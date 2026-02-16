# Predictive Alerting System with ML

A machine learning-based system that predicts infrastructure failures before they happen by learning patterns from metrics. Reduces alert fatigue with smart, adaptive thresholds.

## Features

- **Anomaly Detection**: Learns normal behavior patterns and detects deviations
- **Predictive Alerts**: Forecasts potential failures 5-30 minutes in advance
- **Smart Thresholds**: Dynamic thresholds that adapt to daily/weekly patterns
- **Multi-metric Correlation**: Detects issues by correlating multiple metrics
- **Alert Fatigue Reduction**: Suppresses noisy alerts using confidence scoring
- **Self-learning**: Continuously improves from feedback and new data

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Metrics    │───▶│   Feature    │───▶│   ML Models │
│  Sources    │    │   Engineering│    │  (Isolation │
│(Prometheus, │    │              │    │   Forest,   │
│  InfluxDB)  │    │ - Rolling    │    │   LSTM)     │
└─────────────┘    │   stats      │    └──────┬──────┘
                   │ - Seasonal   │           │
                   │   decomp     │           ▼
                   │ - Rate of    │    ┌─────────────┐
                   │   change     │    │   Alert     │
                   └──────────────┘    │   Engine    │
                                        │             │
                                        │ - Confidence│
                   ┌──────────────┐    │   scoring   │
                   │   Feedback   │◀───│ - Severity  │
                   │   Loop       │    │   levels    │
                   └──────────────┘    └──────┬──────┘
                                              │
                                              ▼
                                        ┌─────────────┐
                                        │  Output     │
                                        │ (Webhooks,  │
                                        │  PagerDuty, │
                                        │  Slack)     │
                                        └─────────────┘
```

## Components

### 1. Data Collector (`collector.py`)
Fetches metrics from various sources (Prometheus, InfluxDB, files)

### 2. Feature Engineering (`features.py`)
Transforms raw metrics into ML-ready features:
- Rolling statistics (mean, std, min, max)
- Rate of change
- Seasonal decomposition
- Time-based features (hour, day of week)

### 3. ML Models (`models.py`)
- **Isolation Forest**: Unsupervised anomaly detection
- **LSTM Autoencoder**: Deep learning for complex patterns
- **Prophet**: Time series forecasting for predictions

### 4. Alert Engine (`alerter.py`)
- Confidence scoring (0-100%)
- Severity classification (info, warning, critical)
- Alert suppression and grouping
- Root cause hints

### 5. API Server (`api.py`)
REST API for querying predictions and managing the system

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Start the system with sample data
python -m predictive_alerting --config config.yaml

# Or run components separately
python collector.py --source prometheus --url http://localhost:9090
python train.py --data metrics.csv
python predict.py --metric cpu_usage
```

### Configuration

```yaml
# config.yaml
sources:
  prometheus:
    url: http://localhost:9090
    metrics:
      - cpu_usage
      - memory_usage
      - disk_io

models:
  anomaly_detection:
    type: isolation_forest
    contamination: 0.01
  
  forecasting:
    type: prophet
    horizon_minutes: 30

alerting:
  min_confidence: 0.8
  channels:
    - type: webhook
      url: https://hooks.slack.com/...
    - type: pagerduty
      key: your-integration-key
```

## API Endpoints

- `GET /health` - Health check
- `GET /predictions` - List current predictions
- `GET /predictions/{metric}` - Predictions for specific metric
- `POST /feedback` - Provide feedback on alerts
- `GET /metrics` - System metrics

## Testing

```bash
pytest tests/
```

## License

MIT
