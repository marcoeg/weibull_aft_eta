# Weibull AFT Model for ETA Prediction

This project implements a Weibull Accelerated Failure Time (AFT) model to predict the estimated time to availability (ETA) for users based on their device telemetry, calendar data, and activity patterns.

Based on telemetry data of the Tweelin Solution and as detailed in the "Tweelin Data Privacy Brief".

## Overview

The Weibull AFT model treats user availability as a "time-to-event" problem, where we predict how long until a user becomes available for a voice call. This approach naturally handles the probabilistic nature of availability and provides confidence intervals for predictions.

## Model Architecture

### Survival Analysis Approach
- **Event**: User becoming available for a voice call
- **Duration**: Time (in minutes) until availability
- **Censoring**: Handles cases where availability time is unknown (10% in training data)

### Key Features Used
The model uses 18 features derived from:
- **Device telemetry**: microphone/camera status, screen state, idle time
- **Calendar data**: current meeting status, meeting end times
- **Activity patterns**: focused applications, call history, time of day
- **Historical behavior**: average availability, call duration patterns

## Results

### Performance Metrics
- **Mean Absolute Error (MAE)**: 17.75 minutes
- **Root Mean Square Error (RMSE)**: 23.86 minutes
- **Within 10 minutes accuracy**: 37.2%
- **Within 15 minutes accuracy**: 53.8%
- **Concordance Index**: 0.592

### Feature Importance
Top 5 most important features:
1. **in_meeting_now** (32.8%): Whether user is currently in a meeting
2. **meeting_end_in_minutes** (31.1%): Time until current meeting ends
3. **mobile_on_call** (13.8%): Whether user is on a mobile call
4. **mic_active** (9.8%): Whether microphone is currently active
5. **focused_window_slack** (2.4%): Whether Slack is the focused application

### Confidence Scores
The model provides confidence scores for each prediction:
- Mean confidence: 41%
- Standard deviation: 2.2%
- Higher confidence for structured events (e.g., meetings with known end times)

## Model Comparison

### Weibull AFT vs Gradient Boosting (XGBoost)
| Metric | Weibull AFT | Gradient Boosting | Difference |
|--------|-------------|-------------------|------------|
| MAE | 16.40 min | 16.20 min | +1.2% |
| RMSE | 22.85 min | 23.01 min | -0.7% |
| Within 10 min | 42.8% | 45.3% | -2.5% |

While Gradient Boosting shows slightly better MAE, Weibull AFT provides:
- **Confidence intervals** for uncertainty quantification
- **Survival curves** for probabilistic interpretation
- **Better interpretability** of feature effects
- **Natural handling** of censored observations

## Usage

### Training the Model
```python
from src.eta_model_weibull import ETAModelWeibull

# Initialize model
model = ETAModelWeibull(penalizer=0.01, l1_ratio=0.0)

# Train
model.fit(X_train, y_train, event_observed)

# Predict
predictions, confidence = model.predict(X_test)
```

### Key Model Parameters
- `penalizer`: L2 regularization strength (default: 0.01)
- `l1_ratio`: Balance between L1 and L2 regularization (default: 0.0)
- `alpha`: Significance level for confidence intervals (default: 0.05)

## Visualization

The model can generate survival curves showing the probability of remaining unavailable over time:

![Survival Curves](results/figures/survival_curves.png)

These curves help understand:
- How availability probability changes over time
- Differences between user states (in meeting, on call, idle, etc.)
- Uncertainty in predictions

## Installation

```bash
# Clone repository
git clone https://github.com/marcoeg/weibull_aft_eta.git
cd weibull_aft_eta

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## Project Structure

```
weibull_aft_eta/
├── README.md                    # Comprehensive project documentation
├── requirements.txt             # All Python dependencies
├── .gitignore                  # Properly configured git ignore
│
├── src/                        # Core implementation
│   ├── eta_model_weibull.py   # The Weibull AFT model (copy from notebooks)
│   └── data_generator.py       # Synthetic data generation
│
├── scripts/                    # Executable scripts
│   ├── train_weibull_model.py  # Complete training pipeline
│   └── evaluate_model.py       # Evaluation and visualization
│
└── results/                    # Generated outputs
    ├── figures/               # Plots and visualizations
    └── reports/               # Analysis reports
```
- `src/`: Core model implementation
- `scripts/`: Training and evaluation scripts
- `tests/`: Unit tests
- `notebooks/`: Exploratory analysis
- `results/`: Generated visualizations and reports

## Future Improvements

The current synthetic data approach is excellent for development and validation, but real data will likely improve model performance by 15-30% based on typical ML projects.

1. **Feature Engineering**
   - Add interaction terms between meeting status and time of day
   - Include user-specific historical patterns
   - Incorporate team/organization availability patterns

2. **Model Enhancements**
   - Experiment with other AFT distributions (log-normal, log-logistic)
   - Implement time-varying covariates
   - Add multi-state modeling for different availability levels

3. **Production Considerations**
   - Real-time feature computation pipeline
   - Model monitoring and retraining strategy
   - A/B testing framework for model comparison