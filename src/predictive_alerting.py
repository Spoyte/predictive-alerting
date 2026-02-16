"""
Predictive Alerting System - Main Module
ML-powered infrastructure failure prediction
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

from collector import MetricsCollector
from models import AnomalyDetector, Forecaster
from alerter import AlertEngine
from api import create_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictiveAlertingSystem:
    """Main orchestrator for the predictive alerting system."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.collector = None
        self.anomaly_detector = None
        self.forecaster = None
        self.alert_engine = None
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Predictive Alerting System...")
        
        # Initialize metrics collector
        self.collector = MetricsCollector(self.config.get('sources', {}))
        
        # Initialize ML models
        model_config = self.config.get('models', {})
        self.anomaly_detector = AnomalyDetector(model_config.get('anomaly_detection', {}))
        self.forecaster = Forecaster(model_config.get('forecasting', {}))
        
        # Initialize alert engine
        self.alert_engine = AlertEngine(self.config.get('alerting', {}))
        
        logger.info("System initialized successfully")
    
    def run(self):
        """Run the main prediction loop."""
        logger.info("Starting prediction loop...")
        
        try:
            while True:
                # Collect metrics
                metrics = self.collector.collect()
                
                # Detect anomalies
                anomalies = self.anomaly_detector.detect(metrics)
                
                # Generate forecasts
                forecasts = self.forecaster.predict(metrics)
                
                # Generate alerts
                alerts = self.alert_engine.process(anomalies, forecasts)
                
                # Send alerts
                for alert in alerts:
                    self.alert_engine.send(alert)
                
                # Sleep until next iteration
                import time
                time.sleep(self.config.get('interval_seconds', 60))
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
    
    def train(self, data_path: str = None):
        """Train models on historical data."""
        logger.info("Training models...")
        
        if data_path:
            import pandas as pd
            data = pd.read_csv(data_path)
        else:
            data = self.collector.get_historical_data()
        
        self.anomaly_detector.train(data)
        self.forecaster.train(data)
        
        logger.info("Training complete")


def main():
    parser = argparse.ArgumentParser(description='Predictive Alerting System')
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to config file')
    parser.add_argument('--train', '-t', help='Train models on historical data file')
    parser.add_argument('--api-only', action='store_true', help='Run API server only')
    parser.add_argument('--port', '-p', type=int, default=5000, help='API server port')
    
    args = parser.parse_args()
    
    system = PredictiveAlertingSystem(args.config)
    
    if args.train:
        system.initialize()
        system.train(args.train)
    elif args.api_only:
        app = create_app(system)
        app.run(host='0.0.0.0', port=args.port)
    else:
        system.initialize()
        system.run()


if __name__ == '__main__':
    main()
