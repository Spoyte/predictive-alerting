"""
ML Models for Anomaly Detection and Forecasting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import json

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features from raw metrics."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from time series data."""
        if df.empty or len(df) < 10:
            return pd.DataFrame()
        
        df = df.copy()
        df = df.sort_values('timestamp')
        
        # Basic rolling statistics
        for window in [5, 10, 30]:
            df[f'rolling_mean_{window}'] = df['value'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['value'].rolling(window=window, min_periods=1).std()
            df[f'rolling_min_{window}'] = df['value'].rolling(window=window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df['value'].rolling(window=window, min_periods=1).max()
        
        # Rate of change
        df['rate_of_change'] = df['value'].diff().fillna(0)
        df['pct_change'] = df['value'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Lag features
        for lag in [1, 5, 10]:
            df[f'lag_{lag}'] = df['value'].shift(lag).fillna(method='bfill')
        
        # Drop NaN values from rolling calculations
        df = df.fillna(0)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return [
            'value', 'rate_of_change', 'pct_change',
            'rolling_mean_5', 'rolling_std_5', 'rolling_min_5', 'rolling_max_5',
            'rolling_mean_10', 'rolling_std_10', 'rolling_min_10', 'rolling_max_10',
            'rolling_mean_30', 'rolling_std_30', 'rolling_min_30', 'rolling_max_30',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
            'lag_1', 'lag_5', 'lag_10'
        ]


class AnomalyDetector:
    """ML-based anomaly detection using Isolation Forest."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get('type', 'isolation_forest')
        self.contamination = config.get('contamination', 0.01)
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        
    def train(self, data: pd.DataFrame):
        """Train the anomaly detection model."""
        logger.info("Training anomaly detection model...")
        
        # Extract features
        df = self.feature_engineer.extract_features(data)
        
        if df.empty:
            logger.warning("Not enough data to train model")
            return
        
        feature_cols = self.feature_engineer.get_feature_columns()
        X = df[feature_cols].values
        
        # Scale features
        X_scaled = self.feature_engineer.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.model.fit(X_scaled)
        
        self.is_trained = True
        logger.info("Anomaly detection model trained")
    
    def detect(self, metrics: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""
        if not self.is_trained:
            logger.warning("Model not trained, skipping detection")
            return []
        
        anomalies = []
        
        for metric_name, df in metrics.items():
            if df.empty:
                continue
            
            # Extract features
            df_features = self.feature_engineer.extract_features(df)
            
            if df_features.empty:
                continue
            
            feature_cols = self.feature_engineer.get_feature_columns()
            X = df_features[feature_cols].values
            X_scaled = self.feature_engineer.scaler.transform(X)
            
            # Predict anomalies
            predictions = self.model.predict(X_scaled)
            scores = self.model.decision_function(X_scaled)
            
            # Find anomaly points (prediction == -1)
            anomaly_indices = np.where(predictions == -1)[0]
            
            for idx in anomaly_indices[-5:]:  # Last 5 anomalies
                anomaly_score = abs(scores[idx])
                anomalies.append({
                    'metric': metric_name,
                    'timestamp': df_features.iloc[idx]['timestamp'],
                    'value': df_features.iloc[idx]['value'],
                    'anomaly_score': float(anomaly_score),
                    'confidence': min(100, anomaly_score * 100),
                    'type': 'anomaly'
                })
        
        return anomalies
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.feature_engineer.scaler,
                'config': self.config
            }, f)
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_engineer.scaler = data['scaler']
            self.config = data['config']
            self.is_trained = True


class Forecaster:
    """Time series forecasting for predictive alerts."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get('type', 'prophet')
        self.horizon_minutes = config.get('horizon_minutes', 30)
        self.models: Dict[str, Any] = {}
        self.feature_engineer = FeatureEngineer()
        
    def train(self, data: pd.DataFrame):
        """Train forecasting models."""
        logger.info("Training forecasting models...")
        
        if 'metric' not in data.columns:
            logger.warning("No metric column found, cannot train forecaster")
            return
        
        for metric_name in data['metric'].unique():
            metric_data = data[data['metric'] == metric_name].copy()
            
            if len(metric_data) < 100:
                continue
            
            try:
                if self.model_type == 'prophet':
                    from prophet import Prophet
                    
                    # Prepare data for Prophet
                    prophet_df = metric_data[['timestamp', 'value']].rename(
                        columns={'timestamp': 'ds', 'value': 'y'}
                    )
                    
                    model = Prophet(
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=False
                    )
                    model.fit(prophet_df)
                    
                    self.models[metric_name] = model
                    logger.info(f"Trained Prophet model for {metric_name}")
                    
            except Exception as e:
                logger.error(f"Failed to train model for {metric_name}: {e}")
    
    def predict(self, metrics: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate forecasts for metrics."""
        predictions = []
        
        for metric_name, df in metrics.items():
            if metric_name not in self.models:
                continue
            
            if df.empty or len(df) < 10:
                continue
            
            try:
                model = self.models[metric_name]
                
                # Create future dataframe
                future = model.make_future_dataframe(
                    periods=self.horizon_minutes,
                    freq='min'
                )
                
                forecast = model.predict(future)
                
                # Get the forecast for the horizon
                future_forecast = forecast.tail(self.horizon_minutes)
                
                # Check if any predicted values are concerning
                current_value = df['value'].iloc[-1]
                max_predicted = future_forecast['yhat'].max()
                min_predicted = future_forecast['yhat'].min()
                
                # Simple threshold-based prediction (can be enhanced)
                threshold_high = df['value'].quantile(0.95)
                threshold_low = df['value'].quantile(0.05)
                
                if max_predicted > threshold_high:
                    predictions.append({
                        'metric': metric_name,
                        'predicted_at': future_forecast.iloc[-1]['ds'],
                        'current_value': float(current_value),
                        'predicted_value': float(max_predicted),
                        'confidence': float(future_forecast.iloc[-1]['yhat_upper'] - future_forecast.iloc[-1]['yhat_lower']),
                        'type': 'prediction_high',
                        'minutes_ahead': self.horizon_minutes
                    })
                
                if min_predicted < threshold_low:
                    predictions.append({
                        'metric': metric_name,
                        'predicted_at': future_forecast.iloc[-1]['ds'],
                        'current_value': float(current_value),
                        'predicted_value': float(min_predicted),
                        'confidence': float(future_forecast.iloc[-1]['yhat_upper'] - future_forecast.iloc[-1]['yhat_lower']),
                        'type': 'prediction_low',
                        'minutes_ahead': self.horizon_minutes
                    })
                    
            except Exception as e:
                logger.error(f"Failed to predict for {metric_name}: {e}")
        
        return predictions
