"""
Quick test script to verify the setup works correctly
Run this before starting the dashboard to ensure everything is configured properly
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        import plotly.graph_objects as go
        import plotly.express as px
        from scipy import stats
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def test_modules():
    """Test that our modules can be imported"""
    print("\nTesting custom modules...")
    try:
        from module1_data_ingestion import create_master_dataset, parse_filename_metadata
        from module2_feature_engineering import add_features_to_dataframe, calculate_journey_statistics
        from module3_comparative_analysis import compare_conditions, calculate_correlations
        print("✓ All custom modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Module import error: {e}")
        return False

def test_data_directory():
    """Test that data directory exists"""
    print("\nTesting data directory...")
    data_dir = Path("obd_dataset")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        print(f"✓ Data directory found with {len(csv_files)} CSV files")
        return True
    else:
        print("✗ Data directory 'obd_dataset' not found")
        return False

def test_filename_parsing():
    """Test filename parsing"""
    print("\nTesting filename parsing...")
    try:
        from module1_data_ingestion import parse_filename_metadata
        
        test_cases = [
            "2017-07-14_Seat_Leon_KA_KA_Frei.csv",
            "2017-07-26_Seat_Leon_RT_S_Stau.csv",
            "2018-02-17_Seat_Leon_BB_RT_Normal_Glatteis.csv"
        ]
        
        for filename in test_cases:
            result = parse_filename_metadata(filename)
            print(f"  {filename}")
            print(f"    → Date: {result['date']}, Route: {result['route']}, Condition: {result['condition']}")
        
        print("✓ Filename parsing works correctly")
        return True
    except Exception as e:
        print(f"✗ Filename parsing error: {e}")
        return False

def main():
    print("=" * 60)
    print("OBD-II Analysis System - Setup Test")
    print("=" * 60)
    
    results = []
    results.append(("Package imports", test_imports()))
    results.append(("Module imports", test_modules()))
    results.append(("Data directory", test_data_directory()))
    results.append(("Filename parsing", test_filename_parsing()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! You can now run: streamlit run app.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
