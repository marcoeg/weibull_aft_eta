#!/usr/bin/env python3
"""
Evaluate trained Weibull AFT model and generate visualizations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from src.eta_model_weibull import ETAModelWeibull
from src.data_generator import AlignedDataGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_scenarios(model: ETAModelWeibull):
    """Test model on specific scenarios"""
    scenarios = pd.DataFrame({
        'computer_locked': [0, 0, 0, 1, 0],
        'mic_active': [0, 1, 0, 0, 0],
        'cam_active': [0, 1, 0, 0, 0],
        'screen_share': [0, 0, 0, 0, 0],
        'idle_over_5min': [0, 0, 0, 1, 1],
        'mobile_on_call': [0, 0, 0, 0, 0],
        'in_meeting_now': [0, 1, 0, 0, 0],
        'meeting_end_in_minutes': [0, 25, 0, 0, 0],
        'is_recurring': [0, 1, 0, 0, 0],
        'meeting_is_private': [1, 1, 1, 1, 1],
        'hour_of_day': [10, 14, 11, 16, 9],
        'day_of_week': [1, 3, 2, 4, 0],
        'minutes_since_last_activity': [5, 2, 10, 30, 45],
        'meetings_in_next_hour': [0, 0, 1, 0, 2],
        'avg_availability_this_hour': [0.6, 0.3, 0.7, 0.4, 0.5],
        'calls_accepted_today': [3, 5, 2, 1, 0],
        'avg_call_duration': [15, 12, 20, 18, 10],
        'focused_window_teams': [0, 1, 0, 0, 0],
        'focused_window_zoom': [0, 0, 0, 0, 0],
        'focused_window_slack': [1, 0, 0, 0, 0]
    })
    
    scenario_names = [
        "Active Slack user",
        "In meeting with Teams (25 min left)",
        "Available at desk",
        "Computer locked, idle",
        "Idle for 45 minutes"
    ]
    
    predictions, confidence = model.predict(scenarios, return_confidence=True)
    
    print("\nScenario Predictions:")
    print("-" * 60)
    for i, name in enumerate(scenario_names):
        print(f"{name}:")
        print(f"  Predicted ETA: {predictions[i]:.1f} minutes")
        print(f"  Confidence: {confidence[i]:.1%}")
        print()


def plot_feature_importance(model: ETAModelWeibull):
    """Plot feature importance"""
    importance = model.get_feature_importance(top_n=15)
    
    plt.figure(figsize=(10, 8))
    features = list(importance.keys())
    values = list(importance.values())
    
    plt.barh(features[::-1], values[::-1])
    plt.xlabel('Importance Score')
    plt.title('Top 15 Feature Importance - Weibull AFT Model')
    plt.tight_layout()
    
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/feature_importance.png', dpi=150)
    logger.info("Feature importance plot saved to results/figures/feature_importance.png")
    plt.close()


def plot_survival_curves(model: ETAModelWeibull):
    """Generate survival curves for different user states"""
    # Generate sample data
    gen = AlignedDataGenerator(n_samples=100, seed=999)
    X, _, _ = gen.generate_synthetic_data()
    
    # Select interesting samples
    sample_indices = []
    sample_labels = []
    
    # Find samples for each scenario
    # 1. In meeting with long duration
    mask = (X['in_meeting_now'] == 1) & (X['meeting_end_in_minutes'] > 40)
    if mask.any():
        sample_indices.append(X[mask].index[0])
        sample_labels.append('Long meeting (>40 min)')
    
    # 2. Quick Slack chat
    mask = (X['focused_window_slack'] == 1) & (X['in_meeting_now'] == 0)
    if mask.any():
        sample_indices.append(X[mask].index[0])
        sample_labels.append('Slack conversation')
    
    # 3. Idle user
    mask = (X['idle_over_5min'] == 1) & (X['mic_active'] == 0)
    if mask.any():
        sample_indices.append(X[mask].index[0])
        sample_labels.append('Idle user')
    
    # 4. Active call
    mask = (X['mic_active'] == 1) & (X['in_meeting_now'] == 0)
    if mask.any():
        sample_indices.append(X[mask].index[0])
        sample_labels.append('Ad-hoc call')
    
    if len(sample_indices) > 0:
        # Get survival functions
        X_subset = X.iloc[sample_indices]
        times = np.linspace(0, 60, 61)
        survival_functions = model.predict_survival_function(X_subset, times=times)
        
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(sample_labels):
            survival_functions.iloc[:, i].plot(label=label, linewidth=2)
        
        plt.xlabel('Time (minutes)')
        plt.ylabel('Probability of remaining unavailable')
        plt.title('Availability Survival Curves by User State')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('results/figures/survival_curves.png', dpi=150)
        logger.info("Survival curves saved to results/figures/survival_curves.png")
        plt.close()


def main():
    """Main evaluation pipeline"""
    logger.info("Loading trained model...")
    
    # Load model
    model = ETAModelWeibull()
    model.load('models/weibull_eta_model.pkl')
    
    # Generate test data
    logger.info("Generating test data...")
    gen = AlignedDataGenerator(n_samples=1000, seed=123)
    X_test, y_test, event_test = gen.generate_synthetic_data()
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    metrics = model.evaluate(X_test, y_test, event_test)
    
    print("\nModel Performance on Test Set:")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Test specific scenarios
    evaluate_scenarios(model)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_feature_importance(model)
    plot_survival_curves(model)
    
    # Save evaluation report
    os.makedirs('results/reports', exist_ok=True)
    with open('results/reports/model_evaluation.md', 'w') as f:
        f.write("# Weibull AFT Model Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Test Set Performance\n\n")
        for metric, value in metrics.items():
            f.write(f"- **{metric}**: {value:.3f}\n")
        
        f.write("\n## Feature Importance\n\n")
        f.write("Top 10 most important features:\n\n")
        for feat, imp in model.get_feature_importance(top_n=10).items():
            f.write(f"1. {feat}: {imp:.3f}\n")
    
    logger.info("Evaluation complete! Check results/ directory for outputs.")


if __name__ == "__main__":
    main()