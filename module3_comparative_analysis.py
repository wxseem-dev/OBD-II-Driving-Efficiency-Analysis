"""
Module 3: Comparative Analysis
Purpose: Compare conditions and find patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats


def compare_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by condition and calculate comparative metrics
    
    Args:
        df: DataFrame with condition column and metrics
        
    Returns:
        DataFrame with aggregated statistics per condition
    """
    if 'condition' not in df.columns:
        raise ValueError("DataFrame must have 'condition' column")
    
    # Define metrics to compare
    metrics = {
        'fuel_rate_g_per_km': ['mean', 'median', 'std'],
        'Vehicle Speed Sensor [km/h]': ['mean', 'median', 'std'],
        'acceleration_event': ['sum', 'mean'],
        'engine_load': ['mean', 'median', 'std'],
    }
    
    # Filter to only existing columns
    existing_metrics = {k: v for k, v in metrics.items() if k in df.columns}
    
    if not existing_metrics:
        return pd.DataFrame()
    
    # Group by condition
    grouped = df.groupby('condition')
    
    # Calculate statistics
    results = []
    for condition, group_df in grouped:
        result = {'condition': condition, 'journey_count': group_df['filename'].nunique()}
        
        for metric, funcs in existing_metrics.items():
            if metric in group_df.columns:
                for func in funcs:
                    if func == 'sum':
                        value = group_df[metric].sum()
                    elif func == 'mean':
                        value = group_df[metric].mean()
                    elif func == 'median':
                        value = group_df[metric].median()
                    elif func == 'std':
                        value = group_df[metric].std()
                    
                    col_name = f"{metric}_{func}"
                    result[col_name] = value
        
        # Speed band percentages
        if 'speed_band' in group_df.columns:
            speed_bands = group_df['speed_band'].value_counts(normalize=True) * 100
            result['pct_stop_and_go'] = speed_bands.get('Stop-and-go', 0)
            result['pct_urban'] = speed_bands.get('Urban', 0)
            result['pct_highway'] = speed_bands.get('Highway', 0)
            result['pct_fast_highway'] = speed_bands.get('Fast highway', 0)
        
        results.append(result)
    
    return pd.DataFrame(results)


def calculate_percentage_differences(comparison_df: pd.DataFrame, 
                                    baseline_condition: str = 'Frei') -> Dict[str, str]:
    """
    Calculate percentage differences between conditions
    
    Args:
        comparison_df: DataFrame from compare_conditions()
        baseline_condition: Condition to use as baseline for comparison
        
    Returns:
        Dictionary with key findings as strings
    """
    if baseline_condition not in comparison_df['condition'].values:
        # Try to find a suitable baseline
        if 'Frei' in comparison_df['condition'].values:
            baseline_condition = 'Frei'
        elif 'Normal' in comparison_df['condition'].values:
            baseline_condition = 'Normal'
        else:
            baseline_condition = comparison_df['condition'].iloc[0]
    
    baseline = comparison_df[comparison_df['condition'] == baseline_condition].iloc[0]
    findings = []
    
    # Compare fuel rate
    fuel_col = 'fuel_rate_g_per_km_mean'
    if fuel_col in comparison_df.columns:
        for _, row in comparison_df.iterrows():
            if row['condition'] != baseline_condition:
                baseline_fuel = baseline[fuel_col]
                other_fuel = row[fuel_col]
                if pd.notna(baseline_fuel) and pd.notna(other_fuel) and baseline_fuel > 0:
                    pct_diff = ((other_fuel - baseline_fuel) / baseline_fuel) * 100
                    findings.append(
                        f"{row['condition']} conditions {'increased' if pct_diff > 0 else 'decreased'} "
                        f"fuel consumption by {abs(pct_diff):.1f}% vs {baseline_condition}"
                    )
    
    # Compare speed
    speed_col = 'Vehicle Speed Sensor [km/h]_mean'
    if speed_col in comparison_df.columns:
        for _, row in comparison_df.iterrows():
            if row['condition'] != baseline_condition:
                baseline_speed = baseline[speed_col]
                other_speed = row[speed_col]
                if pd.notna(baseline_speed) and pd.notna(other_speed) and baseline_speed > 0:
                    pct_diff = ((other_speed - baseline_speed) / baseline_speed) * 100
                    findings.append(
                        f"{row['condition']} conditions had {abs(pct_diff):.1f}% "
                        f"{'higher' if pct_diff > 0 else 'lower'} average speed vs {baseline_condition}"
                    )
    
    # Compare stop-and-go time
    if 'pct_stop_and_go' in comparison_df.columns:
        for _, row in comparison_df.iterrows():
            if row['condition'] != baseline_condition:
                baseline_stop = baseline['pct_stop_and_go']
                other_stop = row['pct_stop_and_go']
                if pd.notna(baseline_stop) and pd.notna(other_stop):
                    pct_diff = other_stop - baseline_stop
                    findings.append(
                        f"{row['condition']} conditions spent {abs(pct_diff):.1f} percentage points "
                        f"{'more' if pct_diff > 0 else 'less'} time in stop-and-go vs {baseline_condition}"
                    )
    
    return {'findings': findings, 'baseline': baseline_condition}


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for key variables
    
    Args:
        df: DataFrame with sensor and feature columns
        
    Returns:
        Correlation matrix DataFrame
    """
    # Select numeric columns for correlation
    numeric_cols = [
        'Vehicle Speed Sensor [km/h]',
        'Engine RPM [RPM]',
        'Absolute Throttle Position [%]',
        'Air Flow Rate from Mass Flow Sensor [g/s]',
        'fuel_rate_g_per_km',
        'engine_load'
    ]
    
    # Filter to existing columns
    existing_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(existing_cols) < 2:
        return pd.DataFrame()
    
    corr_matrix = df[existing_cols].corr()
    
    return corr_matrix


def statistical_significance_test(df: pd.DataFrame, 
                                  metric: str,
                                  condition1: str,
                                  condition2: str) -> Dict:
    """
    Perform statistical significance test between two conditions
    
    Args:
        df: DataFrame with data
        metric: Column name to test
        condition1: First condition name
        condition2: Second condition name
        
    Returns:
        Dictionary with test results
    """
    if metric not in df.columns or 'condition' not in df.columns:
        return {}
    
    group1 = df[df['condition'] == condition1][metric].dropna()
    group2 = df[df['condition'] == condition2][metric].dropna()
    
    if len(group1) < 2 or len(group2) < 2:
        return {}
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    return {
        'metric': metric,
        'condition1': condition1,
        'condition2': condition2,
        'mean1': group1.mean(),
        'mean2': group2.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def get_journey_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary table of all journeys
    
    Args:
        df: Master DataFrame
        
    Returns:
        DataFrame with one row per journey
    """
    from module2_feature_engineering import calculate_journey_statistics
    
    summaries = []
    for filename in df['filename'].unique():
        journey_df = df[df['filename'] == filename]
        stats = calculate_journey_statistics(journey_df)
        summaries.append(stats)
    
    summary_df = pd.DataFrame(summaries)
    
    # Select and order columns
    preferred_cols = [
        'filename', 'date', 'route', 'condition',
        'duration_minutes', 'distance_km', 'avg_speed', 'max_speed',
        'avg_rpm', 'avg_fuel_rate', 'acceleration_events', 'avg_engine_load'
    ]
    
    available_cols = [col for col in preferred_cols if col in summary_df.columns]
    summary_df = summary_df[available_cols]
    
    return summary_df.sort_values('date')


if __name__ == "__main__":
    # Test the module
    from module1_data_ingestion import create_master_dataset
    from module2_feature_engineering import add_features_to_dataframe
    
    print("Loading and processing data...")
    master_df = create_master_dataset(save_path=None)
    master_df = add_features_to_dataframe(master_df)
    
    print("\nComparing conditions...")
    comparison = compare_conditions(master_df)
    print(comparison)
    
    print("\nKey findings:")
    findings = calculate_percentage_differences(comparison)
    for finding in findings['findings']:
        print(f"  - {finding}")
    
    print("\nCorrelation matrix:")
    corr = calculate_correlations(master_df)
    print(corr)
