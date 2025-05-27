#!/usr/bin/env python3
"""
Train Weibull AFT model for ETA prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split

from src.eta_model_weibull import ETAModelWeibull
from src.data_generator import AlignedDataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    logger.info("Starting Weibull AFT model training...")
    
    # Generate training data
    logger.info("Generating synthetic training data...")
    train_gen = AlignedDataGenerator(n_samples=5000, seed=42)
    X, y, event_observed = train_gen.generate_synthetic_data()
    
    # Split into train/validation/test
    X_temp, X_test, y_temp, y_test, event_temp, event_test = train_test_split(
        X, y, event_observed, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val, event_train, event_val = train_test_split(
        X_temp, y_temp, event_temp, test_size=0.2, random_state=42
    )
    
    logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize and train model
    model = ETAModelWeibull(penalizer=0.01, l1_ratio=0.0)
    logger.info("Training Weibull AFT model...")
    model.fit(X_train, y_train, event_train, show_progress=True)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_metrics = model.evaluate(X_val, y_val, event_val)
    
    logger.info("Validation Results:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test, event_test)
    
    logger.info("Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    # Save model and results
    os.makedirs('models', exist_ok=True)
    model_path = 'models/weibull_eta_model.pkl'
    model.save(model_path)
    logger.info(f"\nModel saved to {model_path}")
    
    # Save metrics
    results = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'Weibull AFT',
        'hyperparameters': {
            'penalizer': model.penalizer,
            'l1_ratio': model.l1_ratio,
            'alpha': model.alpha
        },
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': model.get_feature_importance(top_n=10),
        'data_splits': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
    }
    
    with open('models/model_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training complete!")
    
    # Print top features
    logger.info("\nTop 10 Most Important Features:")
    for feat, imp in model.get_feature_importance(top_n=10).items():
        logger.info(f"  {feat}: {imp:.3f}")


if __name__ == "__main__":
    main()