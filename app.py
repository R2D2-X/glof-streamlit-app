import streamlit as st
import folium
from streamlit_folium import st_folium
from backend import get_dem_statistics, RealTimeGLOFMonitor, prepare_features, load_model, make_prediction
import os

st.set_page_config(page_title="GLOF Monitoring App", layout="wide")
st.title("🧊 Glacial Lake Outburst Flood (GLOF) Monitoring")

st.markdown("""
This app allows users to:
- Enter latitude and longitude coordinates.
- Visualize the location on a map.
- Retrieve Digital Elevation Model (DEM) statistics.
- Fetch satellite-based lake expansion data.
- Get weather data.
- Run machine learning model for GLOF risk prediction.
""")

# Sidebar inputs
st.sidebar.header("Input Coordinates")
latitude = st.sidebar.number_input("Latitude", value=27.98, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=86.92, format="%.6f")

if st.sidebar.button("Analyze"):
    with st.spinner("Fetching data from all endpoints..."):
        monitor = RealTimeGLOFMonitor((latitude, longitude))

        # Satellite + Weather
        realtime_df = monitor.fetch_all_data()

        # DEM stats
        dem_stats = get_dem_statistics(latitude, longitude)

        # Display on map
        st.subheader("📍 Selected Location")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**Latitude:** {latitude}")
            st.write(f"**Longitude:** {longitude}")
        with col2:
            m = folium.Map(location=[latitude, longitude], zoom_start=11)
            folium.Marker([latitude, longitude], tooltip="Selected Point").add_to(m)
            st_folium(m, width=700, height=500)

        # Display Satellite + Weather
        if not realtime_df.empty:
            st.success("✅ Fetched Satellite and Weather Data")
            st.subheader("🛰️ Satellite & Weather Data")
            st.dataframe(realtime_df)

        else:
            st.error("❌ Failed to retrieve satellite/weather data.")

        # Display DEM
        if dem_stats:
            st.success("✅ DEM Statistics Retrieved")
            st.subheader("🌐 DEM Statistics")
            st.json(dem_stats)
        else:
            st.error("❌ Failed to retrieve DEM statistics.")

        # Prediction
        if dem_stats and not realtime_df.empty and "lake_area_km2" in realtime_df.columns:
            try:
                lake_area_km2 = realtime_df["lake_area_km2"].values[0]
                features = prepare_features(lake_area_km2, dem_stats)

                if os.path.exists("best_rf_model.pkl"):
                    model = load_model("best_rf_model.pkl")
                    prediction = make_prediction(model, features)
                    st.success("✅ GLOF Risk Prediction Complete")
                    st.subheader("🔮 Prediction Result")
                    st.write(f"**Predicted Risk Label:** `{prediction}`")
                else:
                    st.warning("⚠️ Model file not found. Skipping prediction.")
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
        else:
            st.warning("⚠️ Insufficient data to perform prediction.")
else:
    st.info("👈 Enter coordinates and click Analyze to begin.")
