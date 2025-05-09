import streamlit as st
import folium
from streamlit_folium import st_folium
from backend import get_dem_statistics

st.set_page_config(page_title="GLOF Monitoring App", layout="wide")
st.title("🧊 Glacial Lake Outburst Flood (GLOF) Monitoring")

st.markdown("""
This app allows users to:
- Enter latitude and longitude coordinates.
- Visualize the location on a map.
- Retrieve Digital Elevation Model (DEM) statistics from Copernicus.
""")

# Sidebar inputs
st.sidebar.header("Input Coordinates")
latitude = st.sidebar.number_input("Latitude", value=27.98, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=86.92, format="%.6f")

if st.sidebar.button("Analyze"):
    st.subheader("📍 Selected Location")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"**Latitude:** {latitude}")
        st.write(f"**Longitude:** {longitude}")

        # Display DEM statistics
        with st.spinner("Fetching DEM statistics..."):
            stats = get_dem_statistics(latitude, longitude)

        if stats:
            st.success("DEM Data Retrieved")
            st.write("### DEM Statistics")
            st.json(stats)
        else:
            st.error("Failed to retrieve DEM data. Check API or input values.")

    with col2:
        # Map Visualization
        m = folium.Map(location=[latitude, longitude], zoom_start=11)
        folium.Marker([latitude, longitude], tooltip="Selected Point").add_to(m)
        st_data = st_folium(m, width=700, height=500)
else:
    st.info("👈 Enter coordinates and click Analyze to start.")
