import streamlit as st
import pandas as pd
from datetime import datetime
from backend import RealTimeGLOFMonitor, download_copernicus_dem, compute_dem_statistics, prepare_features, load_model, make_prediction

# Main app layout
st.title("GLOF Real-Time Monitoring System")

# Input: Coordinates of the lake (this can be adjusted based on your needs)
latitude = st.number_input("Enter Latitude:", value=53.5587)
longitude = st.number_input("Enter Longitude:", value=108.1650)

# Fetch Data Button
if st.button('Fetch Data'):
    monitor = RealTimeGLOFMonitor((latitude, longitude))

    # Fetch Satellite Data
    result = monitor.fetch_all_data()
    if not result.empty:
        st.subheader("Satellite Data")
        st.write(result[['timestamp', 'lake_area_km2', 'expansion_rate_pct', 'precipitation_mm']])

    # Fetch DEM Data
    dem_file = download_copernicus_dem(latitude, longitude, radius_km=1)
    if dem_file:
        stats = compute_dem_statistics(dem_file)
        st.subheader("DEM Statistics")
        for key, value in stats.items():
            st.write(f"{key}: {value}")

        # If valid satellite data is available, make a prediction
        if 'lake_area_km2' in result.columns:
            lake_area_km2 = result['lake_area_km2'].values[0]
            features = prepare_features(lake_area_km2, stats)
            model = load_model('best_rf_model.pkl')  # Ensure the model is available in the correct location
            prediction = make_prediction(model, features)
            st.subheader("Prediction")
            st.write(f"GLOF Risk Prediction: {prediction}")
        else:
            st.write("No valid satellite data available for prediction.")
