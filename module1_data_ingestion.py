"""
Module 1: Data Ingestion
Purpose: Load and prepare all journey data from CSV files
"""

import pandas as pd
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


def parse_filename_metadata(filename: str) -> Dict[str, str]:
    """
    Extract metadata from filename pattern: YYYY-MM-DD_Seat_Leon_ROUTE1_ROUTE2_CONDITION.csv
    Examples:
        - 2017-07-14_Seat_Leon_KA_KA_Frei.csv -> route: KA, condition: Frei
        - 2017-07-26_Seat_Leon_RT_S_Stau.csv -> route: RT, condition: Stau
        - 2018-02-17_Seat_Leon_BB_RT_Normal_Glatteis.csv -> route: BB, condition: Normal_Glatteis
    
    Args:
        filename: CSV filename (with or without path)
        
    Returns:
        Dictionary with date, route, condition, and full filename
    """
    # Extract just the filename if path is included
    basename = os.path.basename(filename)
    
    # Remove .csv extension
    name_without_ext = basename.replace('.csv', '')
    
    # Parse pattern: YYYY-MM-DD_Seat_Leon_ROUTE1_ROUTE2_CONDITION
    parts = name_without_ext.split('_')
    
    if len(parts) >= 5:
        date = parts[0]  # YYYY-MM-DD
        # Route is typically the 3rd part (index 3) after "Seat_Leon"
        route = parts[3] if len(parts) > 3 else "unknown"
        # Condition is everything after the route segments
        # Usually parts[4] or parts[4:] if condition has underscores
        if len(parts) == 5:
            condition = parts[4]
        else:
            # Condition may have underscores (e.g., "Normal_Glatteis", "RT_Frei_Beschleunigung")
            condition = '_'.join(parts[4:])
    else:
        # Fallback parsing
        date = parts[0] if len(parts) > 0 else "unknown"
        route = parts[-2] if len(parts) > 1 else "unknown"
        condition = parts[-1] if len(parts) > 0 else "unknown"
    
    return {
        'date': date,
        'route': route,
        'condition': condition,
        'filename': basename
    }


def load_single_journey(filepath: str) -> pd.DataFrame:
    """
    Load a single CSV file into a pandas DataFrame
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with journey data
    """
    try:
        df = pd.read_csv(filepath)
        
        # Parse metadata from filename
        metadata = parse_filename_metadata(filepath)
        
        # Add metadata columns
        df['date'] = metadata['date']
        df['route'] = metadata['route']
        df['condition'] = metadata['condition']
        df['filename'] = metadata['filename']
        
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


def filter_moving_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove stationary vehicle records (speed = 0) and verify no missing values
    
    Args:
        df: DataFrame with journey data
        
    Returns:
        Filtered DataFrame with only moving data
    """
    # Identify speed column (handle different possible names)
    speed_col = None
    for col in df.columns:
        if 'speed' in col.lower() or 'Speed' in col:
            speed_col = col
            break
    
    if speed_col is None:
        print("Warning: Speed column not found, skipping filter")
        return df
    
    # Filter out stationary data (speed = 0)
    df_moving = df[df[speed_col] > 0].copy()
    
    # Verify no missing values in key sensor columns
    sensor_cols = [
        'Engine RPM [RPM]',
        'Vehicle Speed Sensor [km/h]',
        'Air Flow Rate from Mass Flow Sensor [g/s]',
        'Absolute Throttle Position [%]'
    ]
    
    # Check which columns exist
    existing_sensor_cols = [col for col in sensor_cols if col in df_moving.columns]
    
    if existing_sensor_cols:
        # Remove rows with missing values in sensor columns
        df_moving = df_moving.dropna(subset=existing_sensor_cols)
    
    return df_moving


def load_all_journeys(data_dir: str = "obd_dataset") -> List[pd.DataFrame]:
    """
    Load all CSV files from the specified directory
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        List of DataFrames, one per journey
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    journeys = []
    for csv_file in csv_files:
        print(f"Loading {csv_file.name}...")
        df = load_single_journey(str(csv_file))
        if not df.empty:
            journeys.append(df)
    
    print(f"Successfully loaded {len(journeys)} journeys")
    return journeys


def create_master_dataset(data_dir: str = "obd_dataset", 
                         save_path: str = "master_dataset.pkl") -> pd.DataFrame:
    """
    Create master dataset by loading all journeys, filtering, and concatenating
    
    Args:
        data_dir: Directory containing CSV files
        save_path: Optional path to save the master dataset as pickle
        
    Returns:
        Master DataFrame with all journeys combined
    """
    # Load all journeys
    journeys = load_all_journeys(data_dir)
    
    # Filter each journey
    filtered_journeys = []
    for df in journeys:
        df_filtered = filter_moving_data(df)
        if not df_filtered.empty:
            filtered_journeys.append(df_filtered)
    
    # Concatenate all journeys
    if not filtered_journeys:
        raise ValueError("No valid journey data after filtering")
    
    master_df = pd.concat(filtered_journeys, ignore_index=True)
    
    # Convert Time column to datetime if it exists
    if 'Time' in master_df.columns:
        try:
            master_df['Time'] = pd.to_datetime(master_df['Time'], errors='coerce')
        except:
            pass
    
    # Sort by filename and time
    if 'Time' in master_df.columns:
        master_df = master_df.sort_values(['filename', 'Time']).reset_index(drop=True)
    else:
        master_df = master_df.sort_values('filename').reset_index(drop=True)
    
    print(f"\nMaster dataset created:")
    print(f"  Total rows: {len(master_df):,}")
    print(f"  Total columns: {len(master_df.columns)}")
    print(f"  Unique journeys: {master_df['filename'].nunique()}")
    print(f"  Conditions: {master_df['condition'].unique()}")
    
    # Save if path provided
    if save_path:
        master_df.to_pickle(save_path)
        print(f"\nMaster dataset saved to {save_path}")
    
    return master_df


if __name__ == "__main__":
    # Test the module
    master_df = create_master_dataset()
    print("\nFirst few rows:")
    print(master_df.head())
    print("\nColumn names:")
    print(master_df.columns.tolist())
