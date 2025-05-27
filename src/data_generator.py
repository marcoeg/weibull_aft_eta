"""
Synthetic data generator for model development and testing
"""

import numpy as np
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta, timezone


class AlignedDataGenerator:
    """Generate synthetic data aligned with production patterns"""
    
    def __init__(self, n_samples: int = 1000, n_users: int = 50, seed: int = 42):
        self.n_samples = n_samples
        self.n_users = n_users
        np.random.seed(seed)
        
    def generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Generate synthetic data that matches production patterns
        
        Returns:
            X: Feature dataframe
            y: Duration until availability (minutes)
            event_observed: Whether the event was observed (1) or censored (0)
        """
        # Generate base data
        df = self._generate_base_data()
        
        # Add patterns
        df = self._add_meeting_patterns(df)
        df = self._add_activity_patterns(df)
        df = self._generate_availability_labels(df)
        df = self._generate_eta_targets(df)
        
        # Create feature matrix
        X = self._create_feature_matrix(df)
        
        # Create target and event indicators
        y = self._create_duration_target(df)
        event_observed = pd.Series(
            np.random.binomial(1, 0.9, len(df)), 
            name='event_observed'
        )
        
        return X, y, event_observed
    
    def _generate_base_data(self) -> pd.DataFrame:
        """Generate base telemetry data with realistic patterns"""
        # Generate timestamps across 7 days
        UTC = timezone.utc
        timestamps = [
            datetime.now(UTC) - timedelta(minutes=np.random.randint(0, 7*24*60))
            for _ in range(self.n_samples)
        ]
        
        df = pd.DataFrame({
            'user_id': [f'user_{i % self.n_users}' for i in range(self.n_samples)],
            'timestamp': timestamps,
            'hour_of_day': [ts.hour for ts in timestamps],
            'day_of_week': [ts.weekday() for ts in timestamps],
            'is_weekend': [ts.weekday() >= 5 for ts in timestamps]
        })
        
        # Device type - more computers during work hours
        df['device_type'] = df.apply(
            lambda row: np.random.choice(['computer', 'mobile'], 
                                       p=[0.9, 0.1] if 9 <= row['hour_of_day'] <= 17 and not row['is_weekend'] 
                                       else [0.3, 0.7]),
            axis=1
        )
        
        return df
    
    def _add_meeting_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic meeting patterns"""
        # Meetings more likely during business hours
        meeting_prob = df.apply(
            lambda row: 0.6 if 9 <= row['hour_of_day'] <= 17 and not row['is_weekend'] 
                       else 0.1,
            axis=1
        )
        df['in_meeting_now'] = np.random.binomial(1, meeting_prob)
        
        # Meeting characteristics
        df['is_recurring'] = np.where(
            df['in_meeting_now'] == 1,
            np.random.binomial(1, 0.7),  # Most meetings are recurring
            0
        )
        
        df['meeting_is_private'] = np.where(
            df['in_meeting_now'] == 1,
            np.random.binomial(1, 0.8),
            np.random.binomial(1, 0.5)  # Variation for non-meetings
        )
        
        # Meeting duration
        df['meeting_end_in_minutes'] = 0
        mask_recurring = (df['in_meeting_now'] == 1) & (df['is_recurring'] == 1)
        mask_adhoc = (df['in_meeting_now'] == 1) & (df['is_recurring'] == 0)
        
        df.loc[mask_recurring, 'meeting_end_in_minutes'] = np.random.normal(30, 10, mask_recurring.sum()).clip(10, 60).astype(int)
        df.loc[mask_adhoc, 'meeting_end_in_minutes'] = np.random.normal(45, 15, mask_adhoc.sum()).clip(15, 90).astype(int)
        
        return df
    
    def _add_activity_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic activity patterns based on context"""
        # Focused window
        df['focused_window'] = None
        
        # In meeting - likely to have meeting app focused
        meeting_mask = df['in_meeting_now'] == 1
        df.loc[meeting_mask, 'focused_window'] = np.random.choice(
            ['teams', 'zoom', 'slack', None],
            size=meeting_mask.sum(),
            p=[0.4, 0.35, 0.05, 0.2]
        )
        
        # Not in meeting - more varied
        not_meeting_mask = df['in_meeting_now'] == 0
        df.loc[not_meeting_mask, 'focused_window'] = np.random.choice(
            ['teams', 'zoom', 'slack', None],
            size=not_meeting_mask.sum(),
            p=[0.1, 0.05, 0.35, 0.5]
        )
        
        # Mic active - correlated with meetings
        df['mic_active'] = 0
        
        # High probability if in meeting with meeting app
        mask1 = (df['in_meeting_now'] == 1) & (df['focused_window'].isin(['teams', 'zoom']))
        df.loc[mask1, 'mic_active'] = np.random.binomial(1, 0.8, mask1.sum())
        
        # Medium probability if in meeting without app
        mask2 = (df['in_meeting_now'] == 1) & (~df['focused_window'].isin(['teams', 'zoom']))
        df.loc[mask2, 'mic_active'] = np.random.binomial(1, 0.3, mask2.sum())
        
        # Ad-hoc calls
        mask3 = (df['in_meeting_now'] == 0) & (df['focused_window'].isin(['teams', 'zoom']))
        df.loc[mask3, 'mic_active'] = np.random.binomial(1, 0.4, mask3.sum())
        
        # Low probability otherwise
        mask4 = ~(mask1 | mask2 | mask3)
        df.loc[mask4, 'mic_active'] = np.random.binomial(1, 0.02, mask4.sum())
        
        # Camera active - correlated with mic
        df['cam_active'] = np.where(
            df['mic_active'] == 1,
            np.random.binomial(1, 0.6, len(df)),
            np.random.binomial(1, 0.02, len(df))
        )
        
        # Screen share - only in meetings
        df['screen_share'] = np.where(
            (df['in_meeting_now'] == 1) & (df['mic_active'] == 1),
            np.random.binomial(1, 0.3, len(df)),
            0
        )
        
        # Mobile on call
        df['mobile_on_call'] = 0
        mobile_mask = df['device_type'] == 'mobile'
        df.loc[mobile_mask, 'mobile_on_call'] = np.random.binomial(1, 0.15, mobile_mask.sum())
        computer_mask = df['device_type'] == 'computer'
        df.loc[computer_mask, 'mobile_on_call'] = np.random.binomial(1, 0.02, computer_mask.sum())
        
        # Idle status
        idle_prob = np.where(
            (df['mic_active'] == 1) | (df['in_meeting_now'] == 1),
            0.05,
            0.4
        )
        df['idle_over_5min'] = np.random.binomial(1, idle_prob)
        
        # Additional features
        df['minutes_since_last_activity'] = np.random.exponential(15, len(df))
        df['meetings_in_next_hour'] = np.random.poisson(0.5, len(df))
        df['avg_availability_this_hour'] = np.random.uniform(0.3, 0.8, len(df))
        df['calls_accepted_today'] = np.random.poisson(3, len(df))
        df['avg_call_duration'] = np.random.normal(15, 5, len(df)).clip(5, 45)
        
        # Computer locked
        df['computer_locked'] = np.where(
            df['device_type'] == 'computer',
            np.random.binomial(1, 0.1 if df['mic_active'].any() else 0.3, len(df)),
            0
        )
        
        return df
    
    def _generate_availability_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic availability labels based on activity"""
        df['available'] = 1  # Start with everyone available
        
        # Unavailability rules
        df.loc[df['mobile_on_call'] == 1, 'available'] = 0
        df.loc[df['mic_active'] == 1, 'available'] = 0
        df.loc[(df['in_meeting_now'] == 1) & (df['mic_active'] == 1), 'available'] = 0
        df.loc[df['screen_share'] == 1, 'available'] = 0
        
        # Meeting with app focused
        mask_meeting_focused = (
            (df['in_meeting_now'] == 1) & 
            (df['focused_window'].isin(['teams', 'zoom'])) &
            (df['available'] == 1)
        )
        df.loc[mask_meeting_focused, 'available'] = np.random.binomial(
            1, 0.1, mask_meeting_focused.sum()
        )
        
        # Passive meeting
        mask_passive_meeting = (
            (df['in_meeting_now'] == 1) & 
            (df['mic_active'] == 0) & 
            (~df['focused_window'].isin(['teams', 'zoom'])) &
            (df['available'] == 1)
        )
        df.loc[mask_passive_meeting, 'available'] = np.random.binomial(
            1, 0.3, mask_passive_meeting.sum()
        )
        
        # Slack active
        mask_slack_active = (
            (df['focused_window'] == 'slack') & 
            (df['idle_over_5min'] == 0) &
            (df['available'] == 1)
        )
        df.loc[mask_slack_active, 'available'] = np.random.binomial(
            1, 0.4, mask_slack_active.sum()
        )
        
        # Idle users
        mask_idle = (df['idle_over_5min'] == 1) & (df['available'] == 1)
        df.loc[mask_idle, 'available'] = np.random.binomial(
            1, 0.8, mask_idle.sum()
        )
        
        return df
    
    def _generate_eta_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic ETA values for unavailable users"""
        df['eta_minutes'] = 0.0
        unavailable_mask = df['available'] == 0
        
        for idx in df[unavailable_mask].index:
            row = df.loc[idx]
            
            if row['in_meeting_now'] and row['meeting_end_in_minutes'] > 0:
                # Meeting with known end time
                eta = int(row['meeting_end_in_minutes']) + np.random.normal(0, 5)
            elif row['mobile_on_call']:
                # Phone calls are short
                eta = np.random.exponential(10) + 5
            elif row['mic_active'] and row['focused_window'] in ['teams', 'zoom'] and not row['in_meeting_now']:
                # Ad-hoc video calls
                eta = np.random.normal(25, 10)
            elif row['focused_window'] == 'slack':
                # Slack conversations
                eta = np.random.choice([5, 10, 20, 30], p=[0.4, 0.3, 0.2, 0.1])
                eta += np.random.normal(0, 3)
            else:
                # General unavailability
                eta = np.random.normal(20, 10)
            
            df.loc[idx, 'eta_minutes'] = float(np.clip(eta, 2, 90))
        
        return df
    
    def _create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix for model training"""
        feature_columns = [
            'computer_locked', 'mic_active', 'cam_active', 'screen_share',
            'idle_over_5min', 'mobile_on_call', 'in_meeting_now',
            'meeting_end_in_minutes', 'is_recurring', 'meeting_is_private',
            'hour_of_day', 'day_of_week', 'minutes_since_last_activity',
            'meetings_in_next_hour', 'avg_availability_this_hour',
            'calls_accepted_today', 'avg_call_duration'
        ]
        
        # Add focused window as binary features
        for app in ['teams', 'zoom', 'slack']:
            df[f'focused_window_{app}'] = (df['focused_window'] == app).astype(int)
            feature_columns.append(f'focused_window_{app}')
        
        return df[feature_columns]
    
    def _create_duration_target(self, df: pd.DataFrame) -> pd.Series:
        """Create duration target for survival analysis"""
        y = df['eta_minutes'].copy()
        
        # For available users, generate time until next unavailability
        available_mask = df['available'] == 1
        y[available_mask] = np.random.exponential(30, available_mask.sum()) + 10
        y = y.clip(1, 120)  # Ensure positive and reasonable
        
        return pd.Series(y, name='duration_to_availability')