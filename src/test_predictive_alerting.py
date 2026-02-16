"""
Tests for Predictive Alerting System
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import FeatureEngineer, AnomalyDetector, Forecaster
from alerter import AlertEngine, Alert


class TestFeatureEngineer(unittest.TestCase):
    """Test feature engineering."""
    
    def setUp(self):
        self.engineer = FeatureEngineer()
        
        # Create sample data
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(100)]
        values = np.random.normal(50, 10, 100)
        
        self.df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
    
    def test_extract_features(self):
        """Test feature extraction."""
        features = self.engineer.extract_features(self.df)
        
        self.assertIsNotNone(features)
        self.assertGreater(len(features), 0)
        
        # Check that expected columns exist
        expected_cols = self.engineer.get_feature_columns()
        for col in expected_cols:
            self.assertIn(col, features.columns)
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame(columns=['timestamp', 'value'])
        features = self.engineer.extract_features(empty_df)
        self.assertTrue(features.empty)


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detection."""
    
    def setUp(self):
        self.config = {'type': 'isolation_forest', 'contamination': 0.05}
        self.detector = AnomalyDetector(self.config)
        
        # Create sample training data
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(200)]
        values = np.random.normal(50, 10, 200)
        
        # Inject some anomalies
        values[50:55] += 40
        values[150:155] -= 30
        
        self.training_data = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
    
    def test_train(self):
        """Test model training."""
        self.detector.train(self.training_data)
        self.assertTrue(self.detector.is_trained)
    
    def test_detect(self):
        """Test anomaly detection."""
        self.detector.train(self.training_data)
        
        # Create test metrics
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(50)]
        values = np.random.normal(50, 10, 50)
        values[25:30] += 50  # Anomaly
        
        metrics = {
            'test.metric': pd.DataFrame({
                'timestamp': timestamps,
                'value': values
            })
        }
        
        anomalies = self.detector.detect(metrics)
        self.assertIsInstance(anomalies, list)


class TestAlertEngine(unittest.TestCase):
    """Test alert engine."""
    
    def setUp(self):
        self.config = {
            'min_confidence': 0.8,
            'channels': []
        }
        self.engine = AlertEngine(self.config)
    
    def test_process_anomalies(self):
        """Test processing anomalies into alerts."""
        anomalies = [
            {
                'metric': 'cpu_usage',
                'timestamp': datetime.now(),
                'value': 95.5,
                'anomaly_score': 0.5,
                'confidence': 85.0,
                'type': 'anomaly'
            }
        ]
        
        alerts = self.engine.process(anomalies, [])
        self.assertIsInstance(alerts, list)
        
        if alerts:
            self.assertEqual(alerts[0].metric, 'cpu_usage')
            self.assertEqual(alerts[0].type, 'anomaly')
    
    def test_severity_calculation(self):
        """Test severity calculation."""
        self.assertEqual(self.engine._get_severity(95, 'anomaly'), 'critical')
        self.assertEqual(self.engine._get_severity(85, 'anomaly'), 'warning')
        self.assertEqual(self.engine._get_severity(70, 'anomaly'), 'info')


class TestAlert(unittest.TestCase):
    """Test Alert dataclass."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            id='test-123',
            metric='cpu_usage',
            timestamp=datetime.now(),
            severity='warning',
            confidence=85.0,
            message='Test alert',
            type='anomaly',
            details={'key': 'value'}
        )
        
        self.assertEqual(alert.id, 'test-123')
        self.assertEqual(alert.metric, 'cpu_usage')
    
    def test_alert_to_dict(self):
        """Test converting alert to dict."""
        alert = Alert(
            id='test-123',
            metric='cpu_usage',
            timestamp=datetime.now(),
            severity='warning',
            confidence=85.0,
            message='Test alert',
            type='anomaly',
            details={}
        )
        
        d = alert.to_dict()
        self.assertIn('id', d)
        self.assertIn('metric', d)
        self.assertEqual(d['id'], 'test-123')


if __name__ == '__main__':
    unittest.main()
