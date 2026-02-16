"""
REST API for the Predictive Alerting System
"""

import logging
from flask import Flask, jsonify, request
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def create_app(system=None):
    """Create Flask app for the API."""
    app = Flask(__name__)
    app.system = system
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    @app.route('/predictions', methods=['GET'])
    def get_predictions():
        """Get current predictions."""
        if not app.system or not app.system.forecaster:
            return jsonify({'error': 'System not initialized'}), 503
        
        # Return recent predictions from alert history
        recent = app.system.alert_engine.get_recent_alerts(minutes=60)
        predictions = [a.to_dict() for a in recent if a.type.startswith('prediction')]
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        })
    
    @app.route('/predictions/<metric>', methods=['GET'])
    def get_prediction_for_metric(metric: str):
        """Get predictions for a specific metric."""
        if not app.system or not app.system.alert_engine:
            return jsonify({'error': 'System not initialized'}), 503
        
        recent = app.system.alert_engine.get_recent_alerts(minutes=60)
        predictions = [
            a.to_dict() for a in recent 
            if a.metric == metric and a.type.startswith('prediction')
        ]
        
        return jsonify({
            'metric': metric,
            'predictions': predictions
        })
    
    @app.route('/anomalies', methods=['GET'])
    def get_anomalies():
        """Get recent anomalies."""
        if not app.system or not app.system.alert_engine:
            return jsonify({'error': 'System not initialized'}), 503
        
        recent = app.system.alert_engine.get_recent_alerts(minutes=60)
        anomalies = [a.to_dict() for a in recent if a.type == 'anomaly']
        
        return jsonify({
            'anomalies': anomalies,
            'count': len(anomalies)
        })
    
    @app.route('/alerts', methods=['GET'])
    def get_alerts():
        """Get all recent alerts."""
        if not app.system or not app.system.alert_engine:
            return jsonify({'error': 'System not initialized'}), 503
        
        minutes = request.args.get('minutes', 60, type=int)
        recent = app.system.alert_engine.get_recent_alerts(minutes=minutes)
        
        return jsonify({
            'alerts': [a.to_dict() for a in recent],
            'count': len(recent)
        })
    
    @app.route('/feedback', methods=['POST'])
    def submit_feedback():
        """Submit feedback on an alert."""
        data = request.get_json()
        
        if not data or 'alert_id' not in data:
            return jsonify({'error': 'alert_id is required'}), 400
        
        # Store feedback for model improvement
        feedback = {
            'alert_id': data['alert_id'],
            'was_useful': data.get('was_useful', True),
            'comment': data.get('comment', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Received feedback: {feedback}")
        
        return jsonify({
            'status': 'feedback recorded',
            'feedback': feedback
        })
    
    @app.route('/metrics', methods=['GET'])
    def get_metrics():
        """Get system metrics."""
        if not app.system:
            return jsonify({'error': 'System not initialized'}), 503
        
        return jsonify({
            'system': {
                'initialized': app.system.anomaly_detector is not None,
                'models_trained': (
                    app.system.anomaly_detector.is_trained 
                    if app.system.anomaly_detector else False
                ),
                'sources_configured': len(app.system.config.get('sources', {})),
                'channels_configured': len(
                    app.system.config.get('alerting', {}).get('channels', [])
                )
            },
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/config', methods=['GET'])
    def get_config():
        """Get current configuration (sanitized)."""
        if not app.system:
            return jsonify({'error': 'System not initialized'}), 503
        
        config = app.system.config.copy()
        
        # Remove sensitive data
        if 'alerting' in config and 'channels' in config['alerting']:
            for channel in config['alerting']['channels']:
                if 'key' in channel:
                    channel['key'] = '***'
                if 'token' in channel:
                    channel['token'] = '***'
        
        return jsonify(config)
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
