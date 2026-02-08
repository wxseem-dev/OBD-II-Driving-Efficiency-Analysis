"""
Module 4: Streamlit Dashboard
OBD-II Multi-Journey Driving Efficiency Analysis Dashboard
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from pathlib import Path

# Import our modules
from module1_data_ingestion import create_master_dataset
from module2_feature_engineering import add_features_to_dataframe, calculate_journey_statistics
from module3_comparative_analysis import (
    compare_conditions, 
    calculate_percentage_differences,
    calculate_correlations,
    get_journey_summary_table
)

# Page configuration
st.set_page_config(
    page_title="OBD-II Driving Efficiency Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .key-finding {
        background-color: #fff4e6;
        padding: 1rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process master dataset with caching"""
    pkl_path = Path("master_dataset.pkl")
    
    if pkl_path.exists():
        st.info("Loading cached master dataset...")
        master_df = pd.read_pickle(pkl_path)
    else:
        st.info("Creating master dataset from CSV files (this may take a minute)...")
        master_df = create_master_dataset(save_path="master_dataset.pkl")
    
    # Add features if not already present
    if 'fuel_rate_g_per_km' not in master_df.columns:
        master_df = add_features_to_dataframe(master_df)
        master_df.to_pickle("master_dataset.pkl")
    
    return master_df


def main():
    st.markdown('<h1 class="main-header">🚗 OBD-II Multi-Journey Driving Efficiency Analysis</h1>', unsafe_allow_html=True)
    
    # Load data
    master_df = load_data()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", ["Overview", "Journey Analysis", "Condition Comparison", "Detailed Analysis"])
    
    if page == "Overview":
        show_overview(master_df)
    elif page == "Journey Analysis":
        show_journey_analysis(master_df)
    elif page == "Condition Comparison":
        show_condition_comparison(master_df)
    elif page == "Detailed Analysis":
        show_detailed_analysis(master_df)


def show_overview(master_df):
    """Dataset overview page"""
    st.header("📊 Dataset Overview")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Journeys", master_df['filename'].nunique())
    with col2:
        st.metric("Total Data Points", f"{len(master_df):,}")
    with col3:
        st.metric("Conditions", len(master_df['condition'].unique()))
    with col4:
        st.metric("Routes", len(master_df['route'].unique()))
    
    # Conditions breakdown
    st.subheader("Conditions Distribution")
    condition_counts = master_df['condition'].value_counts()
    fig = px.bar(
        x=condition_counts.index,
        y=condition_counts.values,
        labels={'x': 'Condition', 'y': 'Number of Data Points'},
        title="Data Points by Condition"
    )
    st.plotly_chart(fig, width="stretch")
    
    # Journey summary table
    st.subheader("Journey Summary Table")
    summary_df = get_journey_summary_table(master_df)
    st.dataframe(summary_df, width="stretch", height=400)


def show_journey_analysis(master_df):
    """Individual journey analysis page"""
    st.header("🔍 Journey Analysis")
    
    # Journey selector
    journeys = sorted(master_df['filename'].unique())
    selected_journey = st.selectbox("Select Journey", journeys)
    
    # Filter data for selected journey
    journey_df = master_df[master_df['filename'] == selected_journey].copy()
    
    if journey_df.empty:
        st.warning("No data found for selected journey")
        return
    
    # Journey metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Route", journey_df['route'].iloc[0])
    with col2:
        st.metric("Condition", journey_df['condition'].iloc[0])
    
    # Calculate journey statistics
    stats = calculate_journey_statistics(journey_df)
    
    with col3:
        duration = stats.get('duration_minutes', 0)
        st.metric("Duration", f"{duration:.1f} min" if duration else "N/A")
    with col4:
        distance = stats.get('distance_km', 0)
        st.metric("Distance", f"{distance:.2f} km" if distance else "N/A")
    
    # Real-time signals plot
    st.subheader("Real-Time Signals")
    
    # Prepare time axis
    if 'Time' in journey_df.columns:
        try:
            times = pd.to_datetime(journey_df['Time'], errors='coerce')
            if times.notna().any():
                time_axis = (times - times.min()).dt.total_seconds()
            else:
                time_axis = np.arange(len(journey_df)) * 0.1
        except Exception:
            time_axis = np.arange(len(journey_df)) * 0.1
    else:
        time_axis = np.arange(len(journey_df)) * 0.1
    
    # Downsample for plotting to avoid MessageSizeError (max ~2000 points per trace)
    max_points = 2000
    n = len(journey_df)
    if n > max_points:
        step = max(1, n // max_points)
        idx = np.arange(0, n, step)
        if idx[-1] != n - 1:
            idx = np.r_[idx, n - 1]
        time_axis = time_axis.iloc[idx] if hasattr(time_axis, 'iloc') else np.asarray(time_axis)[idx]
        journey_plot = journey_df.iloc[idx].copy()
    else:
        journey_plot = journey_df
    time_axis = np.asarray(time_axis)
    
    # Create subplots with secondary y-axes
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Speed & RPM', 'Throttle Position & MAF'),
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    # Speed (primary y-axis, row 1)
    if 'Vehicle Speed Sensor [km/h]' in journey_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=journey_plot['Vehicle Speed Sensor [km/h]'],
                name='Speed (km/h)',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1,
            secondary_y=False
        )
    
    # RPM (secondary y-axis, row 1)
    if 'Engine RPM [RPM]' in journey_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=journey_plot['Engine RPM [RPM]'],
                name='RPM',
                line=dict(color='red', width=2)
            ),
            row=1, col=1,
            secondary_y=True
        )
    
    # Throttle (primary y-axis, row 2)
    if 'Absolute Throttle Position [%]' in journey_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=journey_plot['Absolute Throttle Position [%]'],
                name='Throttle Position (%)',
                line=dict(color='green', width=2)
            ),
            row=2, col=1,
            secondary_y=False
        )
    
    # MAF (secondary y-axis, row 2)
    if 'Air Flow Rate from Mass Flow Sensor [g/s]' in journey_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=journey_plot['Air Flow Rate from Mass Flow Sensor [g/s]'],
                name='MAF (g/s)',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1,
            secondary_y=True
        )
    
    # Highlight acceleration events (from downsampled plot data)
    if 'acceleration_event' in journey_plot.columns and 'Vehicle Speed Sensor [km/h]' in journey_plot.columns:
        accel_mask = journey_plot['acceleration_event'] == 1
        if accel_mask.any():
            t = np.asarray(time_axis)
            accel_t = t[accel_mask]
            accel_speeds = journey_plot.loc[accel_mask, 'Vehicle Speed Sensor [km/h]'].values
            fig.add_trace(
                go.Scatter(
                    x=accel_t,
                    y=accel_speeds,
                    mode='markers',
                    name='Acceleration Events',
                    marker=dict(color='yellow', size=8, symbol='star')
                ),
                row=1, col=1,
                secondary_y=False
            )
    
    # Update axes
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="RPM", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Throttle (%)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="MAF (g/s)", row=2, col=1, secondary_y=True)
    
    fig.update_layout(
        height=600,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Journey metrics
    st.subheader("Journey Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Speed", f"{stats.get('avg_speed', 0):.1f} km/h")
        st.metric("Max Speed", f"{stats.get('max_speed', 0):.1f} km/h")
    
    with col2:
        st.metric("Average RPM", f"{stats.get('avg_rpm', 0):.0f}")
        st.metric("Fuel Rate", f"{stats.get('avg_fuel_rate', 0):.2f} g/km")
    
    with col3:
        st.metric("Acceleration Events", f"{stats.get('acceleration_events', 0):.0f}")
        st.metric("Engine Load", f"{stats.get('avg_engine_load', 0):.2f}")


def show_condition_comparison(master_df):
    """Condition comparison page"""
    st.header("📈 Condition Comparison")
    
    # Calculate comparison
    comparison_df = compare_conditions(master_df)
    
    if comparison_df.empty:
        st.warning("No comparison data available")
        return
    
    # Display comparison table
    st.subheader("Fuel Consumption by Condition")
    
    # Bar chart for fuel rate
    if 'fuel_rate_g_per_km_mean' in comparison_df.columns:
        fig = px.bar(
            comparison_df,
            x='condition',
            y='fuel_rate_g_per_km_mean',
            error_y='fuel_rate_g_per_km_std' if 'fuel_rate_g_per_km_std' in comparison_df.columns else None,
            labels={'fuel_rate_g_per_km_mean': 'Average Fuel Rate (g/km)', 'condition': 'Condition'},
            title="Average Fuel Consumption by Condition",
            color='fuel_rate_g_per_km_mean',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, width="stretch")
    
    # Key findings
    st.subheader("Key Findings")
    findings_dict = calculate_percentage_differences(comparison_df)
    
    for finding in findings_dict.get('findings', []):
        st.markdown(f'<div class="key-finding">🔍 {finding}</div>', unsafe_allow_html=True)
    
    # Additional comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Vehicle Speed Sensor [km/h]_mean' in comparison_df.columns:
            fig = px.bar(
                comparison_df,
                x='condition',
                y='Vehicle Speed Sensor [km/h]_mean',
                labels={'Vehicle Speed Sensor [km/h]_mean': 'Average Speed (km/h)', 'condition': 'Condition'},
                title="Average Speed by Condition"
            )
            st.plotly_chart(fig, width="stretch")
    
    with col2:
        if 'pct_stop_and_go' in comparison_df.columns:
            fig = px.bar(
                comparison_df,
                x='condition',
                y='pct_stop_and_go',
                labels={'pct_stop_and_go': '% Time in Stop-and-Go', 'condition': 'Condition'},
                title="Stop-and-Go Time by Condition"
            )
            st.plotly_chart(fig, width="stretch")
    
    # Detailed comparison table
    st.subheader("Detailed Comparison Table")
    st.dataframe(comparison_df, width="stretch")


def show_detailed_analysis(master_df):
    """Detailed analysis page with tabs"""
    st.header("🔬 Detailed Analysis")
    
    tabs = st.tabs(["Speed Distribution", "Correlations", "Raw Data"])
    
    with tabs[0]:
        st.subheader("Speed Distribution by Condition")
        
        # Pre-aggregate histograms (send bin counts, not raw 100k+ points) to avoid MessageSizeError
        speed_col = 'Vehicle Speed Sensor [km/h]'
        if speed_col not in master_df.columns:
            st.warning("Speed column not found.")
        else:
            bins = np.arange(0, 141, 5)  # 0, 5, 10, ... 140 km/h
            conditions = master_df['condition'].unique()
            fig = go.Figure()
            colors = px.colors.qualitative.Set3
            for i, condition in enumerate(conditions):
                condition_data = master_df[master_df['condition'] == condition]
                speeds = condition_data[speed_col].dropna()
                if len(speeds) == 0:
                    continue
                counts, bin_edges = np.histogram(speeds, bins=bins)
                # Normalize to % of time
                pct = 100 * counts / counts.sum() if counts.sum() > 0 else counts
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fig.add_trace(go.Bar(
                    x=bin_centers,
                    y=pct,
                    name=condition,
                    opacity=0.7,
                    marker_color=colors[i % len(colors)]
                ))
            fig.update_layout(
                title="Speed Distribution (% of time in each speed band)",
                xaxis_title="Speed (km/h)",
                yaxis_title="% of journey time",
                barmode='overlay',
                height=500
            )
            st.plotly_chart(fig, width="stretch")
    
    with tabs[1]:
        st.subheader("Correlation Analysis")
        
        corr_matrix = calculate_correlations(master_df)
        
        if not corr_matrix.empty:
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            st.plotly_chart(fig, width="stretch")
            
            # Scatter plots
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Vehicle Speed Sensor [km/h]' in master_df.columns and 'fuel_rate_g_per_km' in master_df.columns:
                    fig = px.scatter(
                        master_df.sample(n=min(5000, len(master_df)), random_state=42),
                        x='Vehicle Speed Sensor [km/h]',
                        y='fuel_rate_g_per_km',
                        color='condition',
                        labels={'Vehicle Speed Sensor [km/h]': 'Speed (km/h)', 'fuel_rate_g_per_km': 'Fuel Rate (g/km)'},
                        title="Speed vs Fuel Rate"
                    )
                    st.plotly_chart(fig, width="stretch")
            
            with col2:
                if 'Engine RPM [RPM]' in master_df.columns and 'Air Flow Rate from Mass Flow Sensor [g/s]' in master_df.columns:
                    fig = px.scatter(
                        master_df.sample(n=min(5000, len(master_df)), random_state=42),
                        x='Engine RPM [RPM]',
                        y='Air Flow Rate from Mass Flow Sensor [g/s]',
                        color='condition',
                        labels={'Engine RPM [RPM]': 'RPM', 'Air Flow Rate from Mass Flow Sensor [g/s]': 'MAF (g/s)'},
                        title="RPM vs MAF"
                    )
                    st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Insufficient data for correlation analysis")
    
    with tabs[2]:
        st.subheader("Raw Data Table")
        
        # Allow filtering (use mask + indices to avoid allocating full filtered copy)
        conditions_filter = st.multiselect(
            "Filter by Condition",
            options=master_df['condition'].unique(),
            default=master_df['condition'].unique()
        )
        
        mask = master_df['condition'].isin(conditions_filter)
        total_rows = int(mask.sum())
        max_display_rows = 5000
        display_indices = np.flatnonzero(mask)[:max_display_rows]
        display_df = master_df.iloc[display_indices]
        
        if total_rows > max_display_rows:
            st.info(f"Showing first **{max_display_rows:,}** of **{total_rows:,}** rows. Use the download buttons below for CSV export.")
        
        st.dataframe(
            display_df,
            width="stretch",
            height=400
        )
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_displayed = display_df.to_csv(index=False)
            st.download_button(
                label="Download displayed rows as CSV",
                data=csv_displayed,
                file_name="obd_displayed_rows.csv",
                mime="text/csv"
            )
        with col_dl2:
            # Chunked export: never hold full filtered dataframe in memory
            chunk_size = 100_000
            buf = io.StringIO()
            first = True
            for start in range(0, len(master_df), chunk_size):
                chunk = master_df.iloc[start : start + chunk_size]
                chunk_filtered = chunk.loc[chunk["condition"].isin(conditions_filter)]
                if chunk_filtered.empty:
                    continue
                buf.write(chunk_filtered.to_csv(index=False, header=first))
                first = False
            csv_full = buf.getvalue()
            st.download_button(
                label="Download full filtered data as CSV",
                data=csv_full,
                file_name="obd_filtered_data.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
