"""
Alert Engine - Processes predictions and sends alerts
"""

import logging
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Represents an alert."""
    id: str
    metric: str
    timestamp: datetime
    severity: str  # info, warning, critical
    confidence: float  # 0-100
    message: str
    type: str  # anomaly, prediction_high, prediction_low
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'metric': self.metric,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'confidence': self.confidence,
            'message': self.message,
            'type': self.type,
            'details': self.details
        }


class AlertChannel(ABC):
    """Abstract base class for alert channels."""
    
    @abstractmethod
    def send(self, alert: Alert):
        pass


class WebhookChannel(AlertChannel):
    """Send alerts via webhook."""
    
    def __init__(self, url: str, headers: Optional[Dict] = None):
        self.url = url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    def send(self, alert: Alert):
        try:
            payload = alert.to_dict()
            response = requests.post(self.url, json=payload, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Alert sent to webhook: {alert.id}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")


class SlackChannel(AlertChannel):
    """Send alerts to Slack."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send(self, alert: Alert):
        try:
            color = {
                'info': '#36a64f',
                'warning': '#ff9900',
                'critical': '#ff0000'
            }.get(alert.severity, '#36a64f')
            
            emoji = {
                'anomaly': '🔍',
                'prediction_high': '📈',
                'prediction_low': '📉'
            }.get(alert.type, 'ℹ️')
            
            payload = {
                'attachments': [{
                    'color': color,
                    'title': f"{emoji} {alert.message}",
                    'fields': [
                        {'title': 'Metric', 'value': alert.metric, 'short': True},
                        {'title': 'Severity', 'value': alert.severity.upper(), 'short': True},
                        {'title': 'Confidence', 'value': f"{alert.confidence:.1f}%", 'short': True},
                        {'title': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                    ],
                    'footer': 'Predictive Alerting System'
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Alert sent to Slack: {alert.id}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")


class PagerDutyChannel(AlertChannel):
    """Send alerts to PagerDuty."""
    
    def __init__(self, integration_key: str):
        self.integration_key = integration_key
        self.api_url = 'https://events.pagerduty.com/v2/enqueue'
    
    def send(self, alert: Alert):
        try:
            severity_map = {
                'info': 'info',
                'warning': 'warning',
                'critical': 'critical'
            }
            
            payload = {
                'routing_key': self.integration_key,
                'event_action': 'trigger',
                'dedup_key': alert.id,
                'payload': {
                    'summary': alert.message,
                    'severity': severity_map.get(alert.severity, 'warning'),
                    'source': alert.metric,
                    'custom_details': alert.details
                }
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            logger.info(f"Alert sent to PagerDuty: {alert.id}")
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")


class AlertEngine:
    """Main alert processing engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_confidence = config.get('min_confidence', 0.8)
        self.channels: List[AlertChannel] = []
        self.alert_history: List[Alert] = []
        self._init_channels()
        
    def _init_channels(self):
        """Initialize alert channels from config."""
        for channel_config in self.config.get('channels', []):
            channel_type = channel_config.get('type')
            
            if channel_type == 'webhook':
                self.channels.append(WebhookChannel(
                    channel_config['url'],
                    channel_config.get('headers')
                ))
            elif channel_type == 'slack':
                self.channels.append(SlackChannel(channel_config['url']))
            elif channel_type == 'pagerduty':
                self.channels.append(PagerDutyChannel(channel_config['key']))
            else:
                logger.warning(f"Unknown channel type: {channel_type}")
    
    def process(self, anomalies: List[Dict], predictions: List[Dict]) -> List[Alert]:
        """Process anomalies and predictions into alerts."""
        alerts = []
        
        # Process anomalies
        for anomaly in anomalies:
            if anomaly['confidence'] < self.min_confidence * 100:
                continue
            
            severity = self._get_severity(anomaly['confidence'], 'anomaly')
            alert = Alert(
                id=f"anomaly-{anomaly['metric']}-{anomaly['timestamp'].timestamp()}",
                metric=anomaly['metric'],
                timestamp=anomaly['timestamp'],
                severity=severity,
                confidence=anomaly['confidence'],
                message=f"Anomaly detected in {anomaly['metric']}: value={anomaly['value']:.2f}",
                type='anomaly',
                details={'anomaly_score': anomaly.get('anomaly_score', 0)}
            )
            alerts.append(alert)
        
        # Process predictions
        for prediction in predictions:
            if prediction['confidence'] > 50:  # High uncertainty
                continue
            
            severity = self._get_severity(80, prediction['type'])
            direction = "high" if prediction['type'] == 'prediction_high' else "low"
            
            alert = Alert(
                id=f"pred-{prediction['metric']}-{prediction['predicted_at']}",
                metric=prediction['metric'],
                timestamp=prediction['predicted_at'],
                severity=severity,
                confidence=80,
                message=f"Predicted {direction} value in {prediction['minutes_ahead']}min: {prediction['predicted_value']:.2f}",
                type=prediction['type'],
                details={
                    'current_value': prediction['current_value'],
                    'predicted_value': prediction['predicted_value'],
                    'minutes_ahead': prediction['minutes_ahead']
                }
            )
            alerts.append(alert)
        
        # Deduplicate and filter
        alerts = self._deduplicate(alerts)
        
        return alerts
    
    def _get_severity(self, confidence: float, alert_type: str) -> str:
        """Determine alert severity based on confidence and type."""
        if confidence >= 95:
            return 'critical'
        elif confidence >= 80:
            return 'warning'
        return 'info'
    
    def _deduplicate(self, alerts: List[Alert]) -> List[Alert]:
        """Remove duplicate alerts for the same metric."""
        seen_metrics = set()
        unique_alerts = []
        
        for alert in alerts:
            key = f"{alert.metric}-{alert.type}"
            if key not in seen_metrics:
                seen_metrics.add(key)
                unique_alerts.append(alert)
        
        return unique_alerts
    
    def send(self, alert: Alert):
        """Send alert through all configured channels."""
        self.alert_history.append(alert)
        
        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(f"Failed to send alert through channel: {e}")
    
    def get_recent_alerts(self, minutes: int = 60) -> List[Alert]:
        """Get recent alerts from history."""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self.alert_history if a.timestamp > cutoff]
