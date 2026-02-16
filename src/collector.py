"""
Metrics Collector - Fetches metrics from various sources
"""

import logging
import requests
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MetricsSource(ABC):
    """Abstract base class for metrics sources."""
    
    @abstractmethod
    def fetch(self, metric_name: str, start: datetime, end: datetime) -> pd.DataFrame:
        pass


class PrometheusSource(MetricsSource):
    """Prometheus metrics source."""
    
    def __init__(self, url: str, auth: Optional[Dict] = None):
        self.url = url.rstrip('/')
        self.auth = auth
        
    def fetch(self, metric_name: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch metrics from Prometheus."""
        query_url = f"{self.url}/api/v1/query_range"
        
        params = {
            'query': metric_name,
            'start': start.timestamp(),
            'end': end.timestamp(),
            'step': '60s'
        }
        
        response = requests.get(query_url, params=params, auth=self.auth)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] != 'success':
            raise ValueError(f"Prometheus query failed: {data}")
        
        # Convert to DataFrame
        results = data['data']['result']
        if not results:
            return pd.DataFrame(columns=['timestamp', 'value'])
        
        timestamps = []
        values = []
        
        for result in results[0]['values']:
            timestamps.append(datetime.fromtimestamp(result[0]))
            values.append(float(result[1]))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })


class InfluxDBSource(MetricsSource):
    """InfluxDB metrics source."""
    
    def __init__(self, url: str, token: str, org: str, bucket: str):
        from influxdb_client import InfluxDBClient
        
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        
    def fetch(self, metric_name: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch metrics from InfluxDB."""
        query_api = self.client.query_api()
        
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: {start.isoformat()}, stop: {end.isoformat()})
            |> filter(fn: (r) => r._measurement == "{metric_name}")
            |> aggregateWindow(every: 1m, fn: mean)
        '''
        
        result = query_api.query_data_frame(query)
        
        if result.empty:
            return pd.DataFrame(columns=['timestamp', 'value'])
        
        return pd.DataFrame({
            'timestamp': result['_time'],
            'value': result['_value']
        })


class FileSource(MetricsSource):
    """File-based metrics source (CSV)."""
    
    def __init__(self, path: str):
        self.path = path
        self.data = pd.read_csv(path, parse_dates=['timestamp'])
        
    def fetch(self, metric_name: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch metrics from file."""
        mask = (self.data['timestamp'] >= start) & (self.data['timestamp'] <= end)
        filtered = self.data.loc[mask]
        
        if metric_name not in filtered.columns:
            return pd.DataFrame(columns=['timestamp', 'value'])
        
        return pd.DataFrame({
            'timestamp': filtered['timestamp'],
            'value': filtered[metric_name]
        })


class MetricsCollector:
    """Main collector that aggregates from multiple sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.sources: Dict[str, MetricsSource] = {}
        self.config = config
        self._init_sources()
        
    def _init_sources(self):
        """Initialize all configured sources."""
        for name, source_config in self.config.items():
            source_type = source_config.get('type', name)
            
            if source_type == 'prometheus':
                self.sources[name] = PrometheusSource(
                    source_config['url'],
                    source_config.get('auth')
                )
            elif source_type == 'influxdb':
                self.sources[name] = InfluxDBSource(
                    source_config['url'],
                    source_config['token'],
                    source_config['org'],
                    source_config['bucket']
                )
            elif source_type == 'file':
                self.sources[name] = FileSource(source_config['path'])
            else:
                logger.warning(f"Unknown source type: {source_type}")
    
    def collect(self, lookback_minutes: int = 60) -> Dict[str, pd.DataFrame]:
        """Collect metrics from all sources."""
        end = datetime.now()
        start = end - timedelta(minutes=lookback_minutes)
        
        metrics = {}
        
        for source_name, source in self.sources.items():
            source_config = self.config.get(source_name, {})
            metric_names = source_config.get('metrics', [])
            
            for metric_name in metric_names:
                try:
                    df = source.fetch(metric_name, start, end)
                    metrics[f"{source_name}.{metric_name}"] = df
                    logger.debug(f"Collected {len(df)} points for {metric_name}")
                except Exception as e:
                    logger.error(f"Failed to fetch {metric_name} from {source_name}: {e}")
        
        return metrics
    
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical data for training."""
        end = datetime.now()
        start = end - timedelta(days=days)
        
        all_data = []
        
        for source_name, source in self.sources.items():
            source_config = self.config.get(source_name, {})
            metric_names = source_config.get('metrics', [])
            
            for metric_name in metric_names:
                try:
                    df = source.fetch(metric_name, start, end)
                    df['metric'] = f"{source_name}.{metric_name}"
                    all_data.append(df)
                except Exception as e:
                    logger.error(f"Failed to fetch historical data: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
