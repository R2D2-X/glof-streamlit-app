import logging
import joblib
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from planetary_computer import sign_url
import requests
import geopandas as gpd
from shapely.geometry import box

logging.basicConfig(level=logging.INFO)

COPERNICUS_DEM_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
MODEL_PATH = "models/rf_model.pkl"

class RealTimeGLOFMonitor:
    def __init__(self, bbox):
        self.bbox = bbox  # (minx, miny, maxx, maxy)
        self.dem_path = "/tmp/dem_copernicus.tif"

    def download_copernicus_dem(self):
        logging.info(f"Querying Copernicus DEM for bounds: {self.bbox}")
        aoi = box(*self.bbox)
        query = {
            "collections": ["cop-dem-glo-30"],
            "bbox": list(self.bbox),
            "limit": 1,
        }

        r = requests.post(COPERNICUS_DEM_URL, json=query)
        items = r.json().get("features", [])
        if not items:
            raise Exception("No DEM found for given bounding box.")

        asset_href = items[0]["assets"]["data"]["href"]
        signed_url = sign_url(asset_href)

        with requests.get(signed_url, stream=True) as dem_req:
            with open(self.dem_path, "wb") as f:
                for chunk in dem_req.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info(f"DEM downloaded and saved at {self.dem_path}")
        return self.dem_path

    def compute_dem_statistics(self):
        with rasterio.open(self.dem_path) as src:
            window = from_bounds(*self.bbox, transform=src.transform)
            elevation = src.read(1, window=window, masked=True)
            elevation_data = elevation.compressed()
            if elevation_data.size == 0:
                raise ValueError("No valid elevation data in DEM.")
            return {
                "zmin_m": float(np.min(elevation_data)),
                "zmax_m": float(np.max(elevation_data)),
                "slope_deg": float(np.std(elevation_data)),  # proxy for slope
            }

def prepare_features(area_km2, zmin_m, zmax_m, slope_deg):
    data = {
        "area_km2": [area_km2],
        "zmin_m": [zmin_m],
        "zmax_m": [zmax_m],
        "slope_deg": [slope_deg],
    }
    df = pd.DataFrame(data)
    if df.isnull().values.any():
        logging.error("Feature dataframe contains NaN values:")
        logging.error(df)
        return None
    return df

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

def make_prediction(model, features):
    try:
        if features is None or features.isnull().values.any():
            raise ValueError("Invalid input features. Contains NaN.")
        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0]) * 100
        return {
            "risk_label": prediction,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {
            "risk_label": "Error",
            "confidence": 0.0
        }
