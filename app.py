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

# Initialize session state variables if they don't exist
if "analysis_performed" not in st.session_state:
    st.session_state["analysis_performed"] = False
if "latitude" not in st.session_state:
    st.session_state["latitude"] = 27.98
if "longitude" not in st.session_state:
    st.session_state["longitude"] = 86.92
if "dem_stats" not in st.session_state:
    st.session_state["dem_stats"] = None

# Sidebar inputs
st.sidebar.header("Input Coordinates")
latitude = st.sidebar.number_input("Latitude", value=st.session_state["latitude"], format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=st.session_state["longitude"], format="%.6f")

if st.sidebar.button("Analyze"):
    st.session_state["analysis_performed"] = True
    st.session_state["latitude"] = latitude
    st.session_state["longitude"] = longitude
    with st.spinner("Fetching DEM statistics..."):
        st.session_state["dem_stats"] = get_dem_statistics(latitude, longitude)
else:
    st.info("👈 Enter coordinates and click Analyze to start.")

if st.session_state["analysis_performed"]:
    st.subheader("📍 Selected Location")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"*Latitude:* {st.session_state['latitude']}")
        st.write(f"*Longitude:* {st.session_state['longitude']}")

        # Display DEM statistics
        if st.session_state["dem_stats"]:
            st.success("DEM Data Retrieved")
            st.write("### DEM Statistics")
            st.json(st.session_state["dem_stats"])
        else:
            st.error("Failed to retrieve DEM data. Check API or input values.")

    with col2:
        # Map Visualization
        m = folium.Map(location=[st.session_state["latitude"], st.session_state["longitude"]], zoom_start=11)
        folium.Marker([st.session_state["latitude"], st.session_state["longitude"]], tooltip="Selected Point").add_to(m)
        st_folium(m, width=700, height=500)