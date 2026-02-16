# Predictive Alerting System

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data for testing
python generate_sample_data.py

# Train models on sample data
python -m predictive_alerting --config config.yaml --train sample_metrics.csv

# Run the system
python -m predictive_alerting --config config.yaml

# Or run API server only
python -m predictive_alerting --config config.yaml --api-only --port 5000
```

## Running Tests

```bash
python test_predictive_alerting.py
```

## API Endpoints

Once running, the API is available at `http://localhost:5000`:

- `GET /health` - Health check
- `GET /predictions` - List predictions
- `GET /anomalies` - List anomalies
- `GET /alerts` - List all alerts
- `POST /feedback` - Submit feedback on alerts

## Configuration

Edit `config.yaml` to customize:
- Metrics sources (Prometheus, InfluxDB, files)
- ML model parameters
- Alert channels (webhook, Slack, PagerDuty)
- Confidence thresholds
