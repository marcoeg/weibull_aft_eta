"""
Weibull Accelerated Failure Time (AFT) Model for ETA Prediction
Predicts time until user is available for a voice call
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from lifelines import WeibullAFTFitter
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETAModelWeibull:
    """
    Weibull AFT model for time-to-availability prediction.
    
    The Weibull distribution is particularly suitable for modeling time-to-event data
    as it can capture various hazard patterns (increasing, decreasing, or constant).
    """
    
    def __init__(self, 
                 penalizer: float = 0.01,  # Increased from 0.1 for better regularization
                 l1_ratio: float = 0.0,
                 alpha: float = 0.05,
                 fit_intercept: bool = True):
        """
        Initialize Weibull AFT model.
        
        Args:
            penalizer: L2 regularization strength (increased default for stability)
            l1_ratio: Balance between L1 and L2 (0 = pure L2, 1 = pure L1)
            alpha: Significance level for confidence intervals
            fit_intercept: Whether to fit an intercept term
        """
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
        self.model = None  # Will be created during fit
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.original_feature_names = None
        self.scaler_feature_names = None  # Features the scaler was fit on
        self.fitted_feature_names = None  # Features actually used after preprocessing
        self.dropped_corr_features = []  # Track dropped correlated features
        self.dropped_var_features = []   # Track dropped zero variance features
        self.feature_importance = None
        
    def _validate_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Validate input data"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        if y is not None:
            if not isinstance(y, pd.Series):
                raise TypeError("y must be a pandas Series")
            if len(X) != len(y):
                raise ValueError("X and y must have the same length")
            if (y <= 0).any():
                warnings.warn("Non-positive durations detected. They will be clipped to 0.1")
                
    def _prepare_survival_data(self, 
                             X: pd.DataFrame, 
                             y: pd.Series, 
                             event_observed: Optional[pd.Series] = None,
                             fit_scaler: bool = True) -> pd.DataFrame:
        """
        Prepare data in lifelines format.
        
        Args:
            X: Feature dataframe
            y: Duration until availability (in minutes)
            event_observed: 1 if event was observed, 0 if censored
            fit_scaler: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            DataFrame ready for lifelines
        """
        # Store original features before any modifications
        if fit_scaler:
            self.original_feature_names = X.columns.tolist()
            self.dropped_corr_features = []
            self.dropped_var_features = []
        
        # During training, find and remove correlated features
        if fit_scaler:
            # Check for high correlation and remove redundant features
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with correlation > 0.95
            high_corr_features = [
                column for column in upper_tri.columns 
                if any(upper_tri[column] > 0.95)
            ]
            
            if high_corr_features:
                logger.warning(f"Removing highly correlated features: {high_corr_features}")
                X = X.drop(columns=high_corr_features)
                self.dropped_corr_features = high_corr_features
                
            # Store features that scaler will be fit on
            self.scaler_feature_names = X.columns.tolist()
        else:
            # During prediction, only drop features that were dropped during training
            if self.dropped_corr_features:
                features_to_drop = [f for f in self.dropped_corr_features if f in X.columns]
                if features_to_drop:
                    logger.debug(f"Dropping features that were removed during training: {features_to_drop}")
                    X = X.drop(columns=features_to_drop)
            
            # Select only features the scaler knows about
            X = X[self.scaler_feature_names]
        
        # Standardize features
        if fit_scaler:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        # Check for zero variance features
        if fit_scaler:
            # During training, find and remove zero variance features
            feature_vars = X_scaled.var()
            zero_var_features = feature_vars[feature_vars < 1e-10].index.tolist()
            if zero_var_features:
                logger.warning(f"Removing zero variance features: {zero_var_features}")
                X_scaled = X_scaled.drop(columns=zero_var_features)
                self.dropped_var_features = zero_var_features
                
            # Store the final feature names that will be used
            self.fitted_feature_names = X_scaled.columns.tolist()
        else:
            # During prediction, only drop features that were dropped during training
            if self.dropped_var_features:
                X_scaled = X_scaled.drop(columns=[f for f in self.dropped_var_features if f in X_scaled.columns])
        
        # Create survival dataframe
        survival_df = X_scaled.copy()
        
        # Ensure positive durations (Weibull requires T > 0)
        survival_df['duration'] = y.clip(lower=0.1)
        
        # Add event observation column
        if event_observed is None:
            # Assume all events are observed (no censoring)
            survival_df['event_observed'] = 1
        else:
            survival_df['event_observed'] = event_observed
            
        # Log transformation for very large durations
        max_duration = 120  # 2 hours max
        if (survival_df['duration'] > max_duration).any():
            n_clipped = (survival_df['duration'] > max_duration).sum()
            logger.warning(f"Clipping {n_clipped} durations to {max_duration} minutes")
            survival_df['duration'] = survival_df['duration'].clip(upper=max_duration)
            
        return survival_df
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            event_observed: Optional[pd.Series] = None,
            show_progress: bool = False) -> 'ETAModelWeibull':
        """
        Train the Weibull AFT model.
        
        Args:
            X: Feature dataframe
            y: Duration until availability (in minutes)
            event_observed: 1 if event was observed, 0 if censored
            show_progress: Whether to show training progress
            
        Returns:
            Self for method chaining
        """
        # Validate inputs
        self._validate_data(X, y)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Prepare survival data
        survival_df = self._prepare_survival_data(X, y, event_observed)
        
        # Create model with current parameters
        self.model = WeibullAFTFitter(
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept
        )
        
        # Fit model
        logger.info("Fitting Weibull AFT model...")
        try:
            self.model.fit(
                survival_df,
                duration_col='duration',
                event_col='event_observed',
                show_progress=show_progress
            )
            self.is_fitted = True
            
            # Extract feature importance (absolute parameter values)
            self._calculate_feature_importance()
            
            # Log model summary
            self._log_model_summary()
            
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            # Try with stronger regularization
            logger.info("Retrying with stronger regularization...")
            self.penalizer = self.penalizer * 10
            self.model = WeibullAFTFitter(
                penalizer=self.penalizer,
                l1_ratio=self.l1_ratio,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept
            )
            self.model.fit(
                survival_df,
                duration_col='duration',
                event_col='event_observed',
                show_progress=show_progress
            )
            self.is_fitted = True
            self._calculate_feature_importance()
            self._log_model_summary()
        
        return self
    
    def _calculate_feature_importance(self) -> None:
        """Calculate feature importance based on parameter magnitudes"""
        params = self.model.params_
        
        feature_importance_dict = {}
        
        try:
            # Parameters are stored as a Series with MultiIndex (param_type, covariate)
            # We want lambda_ parameters (excluding Intercept)
            for idx in params.index:
                if isinstance(idx, tuple) and len(idx) == 2:
                    param_type, covariate = idx
                    # Get lambda parameters for actual features (not intercept)
                    if param_type == 'lambda_' and covariate != 'Intercept' and covariate in self.fitted_feature_names:
                        feature_importance_dict[covariate] = abs(float(params[idx]))
            
            # Normalize importance scores
            total_importance = sum(feature_importance_dict.values())
            if total_importance > 0:
                self.feature_importance = {
                    k: v / total_importance 
                    for k, v in feature_importance_dict.items()
                }
                logger.debug(f"Calculated feature importance: {self.feature_importance}")
            else:
                # If no features found, check if we have any features at all
                logger.warning("No feature parameters found in model")
                logger.debug(f"Model parameters: {params}")
                logger.debug(f"Fitted features: {self.fitted_feature_names}")
                # Use equal weights as fallback
                self.feature_importance = {
                    feat: 1.0 / len(self.fitted_feature_names) 
                    for feat in self.fitted_feature_names
                }
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            logger.debug("Exception details:", exc_info=True)
            # Fallback: equal importance for all features
            self.feature_importance = {
                feat: 1.0 / len(self.fitted_feature_names) 
                for feat in self.fitted_feature_names
            }
        
    def _log_model_summary(self) -> None:
        """Log model training summary"""
        logger.info("\n" + "="*50)
        logger.info("Weibull AFT Model Summary")
        logger.info("="*50)
        logger.info(f"Log-likelihood: {self.model.log_likelihood_:.2f}")
        logger.info(f"Concordance Index: {self.model.concordance_index_:.3f}")
        logger.info(f"AIC: {self.model.AIC_:.2f}")
        
        # Print raw parameters for debugging
        logger.info("\nModel parameters:")
        try:
            params_df = self.model.summary
            logger.info(f"\n{params_df}")
        except:
            logger.info(f"Parameters: {self.model.params_}")
        
        # Top features from importance calculation
        if self.feature_importance:
            logger.info("\nTop 10 Most Important Features:")
            sorted_features = sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            for feat, imp in sorted_features:
                logger.info(f"  {feat}: {imp:.3f}")
    
    def predict(self, 
                X: pd.DataFrame,
                return_confidence: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict ETA with confidence intervals.
        
        Args:
            X: Feature dataframe
            return_confidence: Whether to return confidence scores
            
        Returns:
            Tuple of (predictions, confidence_scores) or just predictions
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, returning default predictions")
            default_pred = np.ones(len(X)) * 25  # Default 25 minutes
            default_conf = np.ones(len(X)) * 0.5  # Default 50% confidence
            return (default_pred, default_conf) if return_confidence else default_pred
        
        # Validate and prepare features
        self._validate_data(X)
        
        # Prepare data - need to ensure we have all original features
        X_prepared = X.copy()
        
        # Add any missing original features with zeros
        for feat in self.original_feature_names:
            if feat not in X_prepared.columns:
                logger.debug(f"Adding missing feature '{feat}' with zeros")
                X_prepared[feat] = 0
        
        # Select only the original features in the right order
        X_prepared = X_prepared[self.original_feature_names]
        
        # Apply same preprocessing as training (don't fit scaler)
        survival_df = self._prepare_survival_data(
            X_prepared, 
            pd.Series(np.ones(len(X))),  # Dummy y for preprocessing
            fit_scaler=False
        )
        
        # Get only the features used in the model (excluding duration and event_observed)
        X_final = survival_df[self.fitted_feature_names]
        
        # Get predictions (expected duration)
        predictions = self.model.predict_expectation(X_final)
        
        if return_confidence:
            try:
                # Get the scale parameter for each prediction
                # This gives us a measure of uncertainty
                scale = self.model.predict_median(X_final) / predictions
                
                # Calculate coefficient of variation as a proxy for uncertainty
                # Higher CV = lower confidence
                cv = 1.0 / scale  # Approximate CV
                
                # Convert to confidence score
                # CV typically ranges from 0.1 to 2.0
                confidence = 1 / (1 + cv)
                confidence = confidence.clip(0.3, 0.95)
                
                # Add some variation based on prediction magnitude
                # Longer predictions are generally less certain
                pred_factor = 1 - (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1)
                confidence = confidence * (0.7 + 0.3 * pred_factor)
                confidence = confidence.clip(0.3, 0.95)
                
            except Exception as e:
                logger.warning(f"Error calculating confidence intervals: {e}")
                # Fallback confidence based on prediction magnitude
                confidence = 1 / (1 + predictions / 50)
                confidence = confidence.clip(0.3, 0.8)
            
            return predictions.values, confidence.values
        
        return predictions.values
    
    def predict_survival_function(self, 
                                 X: pd.DataFrame, 
                                 times: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Predict survival function for given features.
        
        Args:
            X: Feature dataframe
            times: Time points to evaluate (default: 0 to 60 minutes)
            
        Returns:
            DataFrame with survival probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if times is None:
            times = np.linspace(0, 60, 61)  # Every minute up to 60
        
        # Prepare data - need to ensure we have all original features
        X_prepared = X.copy()
        
        # Add any missing original features with zeros
        for feat in self.original_feature_names:
            if feat not in X_prepared.columns:
                logger.debug(f"Adding missing feature '{feat}' with zeros")
                X_prepared[feat] = 0
        
        # Select only the original features in the right order
        X_prepared = X_prepared[self.original_feature_names]
        
        # Apply same preprocessing as training (don't fit scaler)
        survival_df = self._prepare_survival_data(
            X_prepared, 
            pd.Series(np.ones(len(X))),  # Dummy y for preprocessing
            fit_scaler=False
        )
        
        # Get only the features used in the model (excluding duration and event_observed)
        X_final = survival_df[self.fitted_feature_names]
        
        # Get survival functions
        survival_functions = self.model.predict_survival_function(X_final, times=times)
        
        return survival_functions
    
    def evaluate(self, 
                X: pd.DataFrame, 
                y: pd.Series,
                event_observed: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature dataframe
            y: True durations
            event_observed: Event observation indicators
            
        Returns:
            Dictionary of metrics
        """
        predictions, confidence = self.predict(X, return_confidence=True)
        
        # Calculate metrics
        mae = np.mean(np.abs(y - predictions))
        rmse = np.sqrt(np.mean((y - predictions) ** 2))
        
        # Percentage within tolerance windows
        within_5 = (np.abs(y - predictions) <= 5).mean()
        within_10 = (np.abs(y - predictions) <= 10).mean()
        within_15 = (np.abs(y - predictions) <= 15).mean()
        
        # Concordance index on test data
        test_df = self._prepare_survival_data(X, y, event_observed)
        concordance = self.model.score(test_df, scoring_method='concordance_index')
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'within_5_min': within_5,
            'within_10_min': within_10,
            'within_15_min': within_15,
            'concordance_index': concordance,
            'mean_confidence': confidence.mean(),
            'std_confidence': confidence.std()
        }
        
        return metrics
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return (None for all)
            
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if top_n is None:
            return self.feature_importance
            
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        return dict(sorted_features)
    
    def save(self, filepath: str) -> None:
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'original_feature_names': self.original_feature_names,
            'scaler_feature_names': self.scaler_feature_names,
            'fitted_feature_names': self.fitted_feature_names,
            'dropped_corr_features': self.dropped_corr_features,
            'dropped_var_features': self.dropped_var_features,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted,
            'penalizer': self.penalizer,
            'l1_ratio': self.l1_ratio,
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.original_feature_names = model_data.get('original_feature_names', model_data['feature_names'])
        self.scaler_feature_names = model_data.get('scaler_feature_names', self.original_feature_names)
        self.fitted_feature_names = model_data.get('fitted_feature_names', model_data['feature_names'])
        self.dropped_corr_features = model_data.get('dropped_corr_features', [])
        self.dropped_var_features = model_data.get('dropped_var_features', [])
        self.feature_importance = model_data['feature_importance']
        self.is_fitted = model_data['is_fitted']
        self.penalizer = model_data.get('penalizer', 0.01)
        self.l1_ratio = model_data.get('l1_ratio', 0.0)
        self.alpha = model_data.get('alpha', 0.05)
        self.fit_intercept = model_data.get('fit_intercept', True)
        
        logger.info(f"Model loaded from {filepath}")
        
    def plot_survival_curves(self, 
                            X: pd.DataFrame, 
                            sample_indices: List[int] = None,
                            **kwargs) -> None:
        """
        Plot survival curves for sample predictions.
        
        Args:
            X: Feature dataframe
            sample_indices: Indices of samples to plot (default: first 5)
            **kwargs: Additional arguments for plotting
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not installed, cannot plot")
            return
            
        if sample_indices is None:
            sample_indices = list(range(min(5, len(X))))
            
        X_subset = X.iloc[sample_indices]
        survival_functions = self.predict_survival_function(X_subset)
        
        plt.figure(figsize=(10, 6))
        for i, idx in enumerate(sample_indices):
            survival_functions.iloc[:, i].plot(
                label=f'Sample {idx}',
                **kwargs
            )
            
        plt.xlabel('Time (minutes)')
        plt.ylabel('Probability of not being available')
        plt.title('Predicted Availability Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()