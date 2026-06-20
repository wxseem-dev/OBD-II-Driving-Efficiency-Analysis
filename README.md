# OBD-II Multi-Journey Driving Efficiency Analysis

A comprehensive analysis system for real-world automotive OBD-II data from 20+ journeys in different driving conditions. This project quantifies how driving conditions (Normal, Frei, Stau, Glatteis, etc.) impact vehicle efficiency and driving behavior.

<img width="1620" height="1080" alt="image" src="https://github.com/user-attachments/assets/ea186b08-4ac9-41c6-a03a-90d9075b576f" />


## Features

- **Data Pipeline**: Automated loading and processing of multiple OBD-II CSV files
- **Feature Engineering**: Calculates fuel efficiency, acceleration events, speed bands, and engine load metrics
- **Comparative Analysis**: Statistical comparison between different driving conditions
- **Interactive Dashboard**: Streamlit-based web interface with interactive visualizations



## Project Structure

```
obd_project/
├── obd_dataset/              # CSV data files (81 journeys)
├── module1_data_ingestion.py # Data loading and preprocessing
├── module2_feature_engineering.py # Feature calculation
├── module3_comparative_analysis.py # Statistical analysis
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Dashboard

Start the Streamlit dashboard:
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Using Individual Modules

You can also use the modules independently:

```python
from module1_data_ingestion import create_master_dataset
from module2_feature_engineering import add_features_to_dataframe
from module3_comparative_analysis import compare_conditions

# Load data
master_df = create_master_dataset()

# Add features
master_df = add_features_to_dataframe(master_df)

# Compare conditions
comparison = compare_conditions(master_df)
```

## Data Pipeline

<img width="1251" height="862" alt="image" src="https://github.com/user-attachments/assets/3e1345fa-43d6-4072-bcdd-0da825a91ab6" />


### Module 1: Data Ingestion
- Iterates through all CSV files in `obd_dataset/`
- Parses filename metadata (date, route, condition)
- Filters out stationary data (speed = 0)
- Creates master dataset (~800k rows)

### Module 2: Feature Engineering
- **Fuel Efficiency**: `fuel_rate = MAF [g/s] / speed [km/h]` (g/km)
- **Acceleration Events**: Detects throttle delta > 10% in < 1 second
- **Speed Bands**: Classifies into Stop-and-go (0-30), Urban (30-60), Highway (60-90), Fast highway (90+)
- **Engine Load**: `(RPM × throttle_position) / 1000`
- **Journey Statistics**: Duration, distance, average/max speed, etc.

### Module 3: Comparative Analysis
- Groups data by condition (Normal/Stau/Frei)
- Calculates comparative metrics (mean, median, std dev)
- Percentage differences between conditions
- Correlation analysis
- Statistical significance testing

### Module 4: Dashboard
- **Overview**: Dataset statistics and journey summary table
- **Journey Analysis**: Individual journey exploration with real-time signal plots
- **Condition Comparison**: Side-by-side comparison of driving conditions
- **Detailed Analysis**: Speed distribution, correlations, raw data tables

## Dashboard Features

### Overview Page
- Total journeys, data points, conditions, routes
- Condition distribution chart
- Journey summary table

### Journey Analysis Page
- Journey selector dropdown
- Real-time signal plot (Speed, RPM, Throttle, MAF)
- Journey metrics (duration, distance, average speed, fuel rate, etc.)
- Acceleration event highlighting

<img width="1195" height="922" alt="image" src="https://github.com/user-attachments/assets/dda8c8ef-7fed-4e22-b29a-aecd4268f845" />


### Condition Comparison Page
- Fuel consumption comparison bar chart
- Key findings (e.g., "Traffic increased fuel consumption by 52%")
- Average speed and stop-and-go time comparisons
- Detailed comparison table

<img width="1170" height="642" alt="image" src="https://github.com/user-attachments/assets/636d6ae0-86c7-4c2c-af41-fe7347b68d1c" />


### Detailed Analysis Page
- **Speed Distribution**: Histogram showing speed distribution by condition
- **Correlations**: Heatmap and scatter plots of variable relationships
- **Raw Data**: Filterable data table with CSV export

<img width="1207" height="572" alt="image" src="https://github.com/user-attachments/assets/b9a69d5c-4cfb-4009-9011-eac33894e6d2" />


## Data Format

CSV files follow the naming pattern:
```
YYYY-MM-DD_Seat_Leon_ROUTE_CONDITION.csv
```

Example: `2017-07-14_Seat_Leon_KA_KA_Frei.csv`

Each CSV contains columns:
- `Time`: Timestamp
- `Engine Coolant Temperature [°C]`
- `Intake Manifold Absolute Pressure [kPa]`
- `Engine RPM [RPM]`
- `Vehicle Speed Sensor [km/h]`
- `Intake Air Temperature [°C]`
- `Air Flow Rate from Mass Flow Sensor [g/s]` (MAF)
- `Absolute Throttle Position [%]`
- `Ambient Air Temperature [°C]`
- `Accelerator Pedal Position D [%]`
- `Accelerator Pedal Position E [%]`

## Key Metrics

- **Fuel Rate**: Grams per kilometer (lower = more efficient)
- **Acceleration Events**: Count of aggressive acceleration instances
- **Speed Bands**: Percentage of journey time in each speed range
- **Engine Load**: Indicator of engine stress level

## Performance Notes

- First run will process all CSV files and create `master_dataset.pkl` (cached for faster subsequent loads)
- Dashboard uses Streamlit caching for optimal performance
- Large datasets are sampled for scatter plots to maintain interactivity
