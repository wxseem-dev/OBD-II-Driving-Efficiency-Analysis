"""
Module 2: Feature Engineering
Purpose: Calculate derived metrics for analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def calculate_fuel_rate(df: pd.DataFrame) -> pd.Series:
    """
    Calculate fuel consumption rate proxy: MAF [g/s] / speed [km/h]
    Lower values = more efficient
    
    Args:
        df: DataFrame with MAF and speed columns
        
    Returns:
        Series with fuel rate values (g/km)
    """
    maf_col = 'Air Flow Rate from Mass Flow Sensor [g/s]'
    speed_col = 'Vehicle Speed Sensor [km/h]'
    
    if maf_col not in df.columns or speed_col not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    # Calculate fuel rate: g/s / (km/h) = g/km * (h/s) = g/km * (1/3600)
    # Actually: (g/s) / (km/h) = (g/s) * (h/km) = (g/s) * (3600 s / km) = g/km * 3600
    # But we want g/km, so: (g/s) / (km/h) * (h/s) = (g/s) / (km/h) * (1/3600)
    # More simply: (g/s) / (km/h) = (g * h) / (s * km) = (g * 3600s) / (s * km) = 3600 * g/km
    
    # To get g/km: (g/s) / (km/h) / 3600
    speed = df[speed_col]
    maf = df[maf_col]
    
    # Avoid division by zero
    fuel_rate = np.where(speed > 0, maf / speed * 3600, np.nan)
    
    return pd.Series(fuel_rate, index=df.index, name='fuel_rate_g_per_km')


def detect_acceleration_events(df: pd.DataFrame, 
                               throttle_col: str = 'Absolute Throttle Position [%]',
                               threshold: float = 10.0,
                               time_window: float = 1.0) -> pd.Series:
    """
    Detect acceleration events where throttle delta > threshold in time window
    
    Args:
        df: DataFrame with throttle data
        throttle_col: Name of throttle position column
        threshold: Minimum throttle delta to count as acceleration event (%)
        time_window: Time window in seconds to check for delta
        
    Returns:
        Series with 1 for acceleration events, 0 otherwise
    """
    if throttle_col not in df.columns:
        return pd.Series(0, index=df.index, name='acceleration_event')
    
    throttle = df[throttle_col].values
    acceleration_events = np.zeros(len(df))
    
    # Calculate throttle delta over rolling window
    # Since we don't have exact time intervals, use a simple rolling difference
    # Approximate: assume ~0.1s between rows (typical OBD-II sampling rate)
    window_size = max(1, int(time_window / 0.1))
    
    for i in range(window_size, len(df)):
        throttle_delta = throttle[i] - throttle[i - window_size]
        if throttle_delta > threshold:
            acceleration_events[i] = 1
    
    return pd.Series(acceleration_events, index=df.index, name='acceleration_event')


def classify_speed_bands(df: pd.DataFrame, 
                        speed_col: str = 'Vehicle Speed Sensor [km/h]') -> pd.Series:
    """
    Classify speed into bands: Stop-and-go (0-30), Urban (30-60), 
    Highway (60-90), Fast highway (90+)
    
    Args:
        df: DataFrame with speed data
        speed_col: Name of speed column
        
    Returns:
        Series with speed band labels
    """
    if speed_col not in df.columns:
        return pd.Series('Unknown', index=df.index, name='speed_band')
    
    speed = df[speed_col]
    
    conditions = [
        speed <= 30,
        (speed > 30) & (speed <= 60),
        (speed > 60) & (speed <= 90),
        speed > 90
    ]
    choices = ['Stop-and-go', 'Urban', 'Highway', 'Fast highway']
    
    return pd.Series(np.select(conditions, choices, default='Unknown'), 
                    index=df.index, name='speed_band')


def calculate_engine_load(df: pd.DataFrame) -> pd.Series:
    """
    Calculate engine load factor: (RPM × throttle_position) / 1000
    Indicates engine stress
    
    Args:
        df: DataFrame with RPM and throttle data
        
    Returns:
        Series with engine load values
    """
    rpm_col = 'Engine RPM [RPM]'
    throttle_col = 'Absolute Throttle Position [%]'
    
    if rpm_col not in df.columns or throttle_col not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    rpm = df[rpm_col]
    throttle = df[throttle_col]
    
    engine_load = (rpm * throttle) / 1000
    
    return pd.Series(engine_load, index=df.index, name='engine_load')


def calculate_journey_statistics(df: pd.DataFrame, 
                                 journey_name: str = None) -> Dict:
    """
    Calculate journey-level statistics
    
    Args:
        df: DataFrame for a single journey
        journey_name: Optional journey identifier
        
    Returns:
        Dictionary with journey statistics
    """
    stats = {}
    
    speed_col = 'Vehicle Speed Sensor [km/h]'
    rpm_col = 'Engine RPM [RPM]'
    time_col = 'Time'
    
    if speed_col in df.columns:
        stats['avg_speed'] = df[speed_col].mean()
        stats['max_speed'] = df[speed_col].max()
        stats['min_speed'] = df[speed_col].min()
    
    if rpm_col in df.columns:
        stats['avg_rpm'] = df[rpm_col].mean()
        stats['max_rpm'] = df[rpm_col].max()
    
    # Duration calculation
    if time_col in df.columns:
        try:
            times = pd.to_datetime(df[time_col], errors='coerce')
            if times.notna().any():
                duration = (times.max() - times.min()).total_seconds()
                stats['duration_seconds'] = duration
                stats['duration_minutes'] = duration / 60
        except:
            # Fallback: use row count * estimated sampling rate
            stats['duration_seconds'] = len(df) * 0.1  # Assume ~0.1s per row
            stats['duration_minutes'] = stats['duration_seconds'] / 60
    
    # Distance estimate: integrate speed over time
    if speed_col in df.columns:
        # Assume constant time interval (0.1s typical for OBD-II)
        time_interval = 0.1  # seconds
        if 'duration_seconds' in stats:
            time_interval = stats['duration_seconds'] / len(df)
        
        # Convert km/h to m/s, multiply by time interval, sum, convert to km
        speed_ms = df[speed_col] / 3.6  # km/h to m/s
        distance_m = (speed_ms * time_interval).sum()
        stats['distance_km'] = distance_m / 1000
    
    # Fuel rate
    if 'fuel_rate_g_per_km' in df.columns:
        stats['avg_fuel_rate'] = df['fuel_rate_g_per_km'].mean()
        stats['total_fuel_consumed'] = df['fuel_rate_g_per_km'].sum() * (time_interval / 3600) if 'distance_km' in stats else None
    
    # Acceleration events
    if 'acceleration_event' in df.columns:
        stats['acceleration_events'] = df['acceleration_event'].sum()
    
    # Engine load
    if 'engine_load' in df.columns:
        stats['avg_engine_load'] = df['engine_load'].mean()
        stats['max_engine_load'] = df['engine_load'].max()
    
    # Speed band distribution
    if 'speed_band' in df.columns:
        speed_bands = df['speed_band'].value_counts(normalize=True) * 100
        stats['pct_stop_and_go'] = speed_bands.get('Stop-and-go', 0)
        stats['pct_urban'] = speed_bands.get('Urban', 0)
        stats['pct_highway'] = speed_bands.get('Highway', 0)
        stats['pct_fast_highway'] = speed_bands.get('Fast highway', 0)
    
    # Metadata
    if 'condition' in df.columns:
        stats['condition'] = df['condition'].iloc[0]
    if 'route' in df.columns:
        stats['route'] = df['route'].iloc[0]
    if 'date' in df.columns:
        stats['date'] = df['date'].iloc[0]
    if 'filename' in df.columns:
        stats['filename'] = df['filename'].iloc[0]
    
    return stats


def add_features_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to the dataframe
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added feature columns
    """
    df = df.copy()
    
    # Add features
    df['fuel_rate_g_per_km'] = calculate_fuel_rate(df)
    df['acceleration_event'] = detect_acceleration_events(df)
    df['speed_band'] = classify_speed_bands(df)
    df['engine_load'] = calculate_engine_load(df)
    
    return df


if __name__ == "__main__":
    # Test the module
    from module1_data_ingestion import create_master_dataset
    
    print("Loading master dataset...")
    master_df = create_master_dataset(save_path=None)
    
    print("\nAdding features...")
    master_df = add_features_to_dataframe(master_df)
    
    print("\nFeature columns added:")
    print([col for col in master_df.columns if col in ['fuel_rate_g_per_km', 'acceleration_event', 'speed_band', 'engine_load']])
    
    print("\nSample statistics for first journey:")
    first_journey = master_df[master_df['filename'] == master_df['filename'].iloc[0]]
    stats = calculate_journey_statistics(first_journey)
    for key, value in stats.items():
        print(f"  {key}: {value}")
