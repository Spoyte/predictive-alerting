"""
Generate sample metrics data for testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_normal_pattern(n_points=1000, base_value=50, noise=5):
    """Generate normal metric pattern with daily seasonality."""
    t = np.arange(n_points)
    
    # Base value with daily seasonality
    daily_pattern = 10 * np.sin(2 * np.pi * t / 1440)  # 1440 minutes = 1 day
    
    # Weekly pattern (lower on weekends)
    weekly_pattern = 5 * np.sin(2 * np.pi * t / (1440 * 7))
    
    # Random noise
    noise_values = np.random.normal(0, noise, n_points)
    
    values = base_value + daily_pattern + weekly_pattern + noise_values
    return np.clip(values, 0, 100)  # Clip to 0-100 range


def inject_anomalies(values, n_anomalies=10):
    """Inject random anomalies into the data."""
    values = values.copy()
    
    for _ in range(n_anomalies):
        idx = random.randint(100, len(values) - 10)
        anomaly_type = random.choice(['spike', 'drop', 'trend'])
        
        if anomaly_type == 'spike':
            values[idx:idx+5] += random.uniform(30, 50)
        elif anomaly_type == 'drop':
            values[idx:idx+5] -= random.uniform(20, 40)
        elif anomaly_type == 'trend':
            trend = np.linspace(0, random.uniform(20, 40), 20)
            values[idx:idx+20] += trend
    
    return np.clip(values, 0, 100)


def generate_sample_data(output_path='sample_metrics.csv', days=7):
    """Generate sample metrics data."""
    n_points = days * 24 * 60  # Minutes in period
    
    start_time = datetime.now() - timedelta(days=days)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
    
    # Generate different metrics
    data = {
        'timestamp': timestamps,
        'cpu_usage': inject_anomalies(generate_normal_pattern(n_points, 45, 8)),
        'memory_usage': inject_anomalies(generate_normal_pattern(n_points, 60, 5)),
        'disk_io': inject_anomalies(generate_normal_pattern(n_points, 20, 15,)),
        'network_in': inject_anomalies(generate_normal_pattern(n_points, 35, 10)),
        'network_out': inject_anomalies(generate_normal_pattern(n_points, 30, 10)),
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {n_points} data points to {output_path}")
    return df


if __name__ == '__main__':
    generate_sample_data()
