import requests
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
import os
import tempfile
import joblib
import logging
import math
import rioxarray
import rasterio
import openmeteo_requests
import retrying
import stackstac
from pystac_client import Client
import planetary_computer
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType


# Setup logging
logging.basicConfig(level=logging.INFO)

class RealTimeGLOFMonitor:
    def __init__(self, lake_coords: tuple, radius_km: int = 5):
        self.config = SHConfig()
        self.config.sh_client_id = "cd893d06-a7c0-49d4-b243-639a69c59eaf"
        self.config.sh_client_secret = "WJKRl2VHKgpvKn0QI1x8aVQwwXZ7fqy9"

        self.lake_bbox = BBox(
            bbox=[
                lake_coords[1] - radius_km / 111,
                lake_coords[0] - radius_km / 111,
                lake_coords[1] + radius_km / 111,
                lake_coords[0] + radius_km / 111
            ],
            crs=CRS.WGS84
        )
        self.bbox_area = self._calculate_bbox_area()
        self.last_area = None
        self.last_update = None

    def _calculate_bbox_area(self):
        min_lon, min_lat, max_lon, max_lat = self.lake_bbox
        earth_radius = 6371
        lat1 = math.radians(min_lat)
        lat2 = math.radians(max_lat)
        lon1 = math.radians(min_lon)
        lon2 = math.radians(max_lon)
        return (earth_radius ** 2) * abs(math.sin(lat2) - math.sin(lat1)) * abs(lon2 - lon1)

    @retrying.retry(wait_exponential_multiplier=1000, stop_max_attempt_number=3)
    def get_sentinel2_ndwi(self) -> dict:
        ndwi_script = """
        //VERSION=3
        function setup() {
            return {
                input: ["B03", "B08", "SCL"],
                output: { bands: 1 }
            };
        }
        function evaluatePixel(sample) {
            let ndwi = index(sample.B03, sample.B08);
            if ([3, 8, 9, 10].includes(sample.SCL)) {
                return [NaN];
            }
            return [ndwi];
        }
        """

        current_time = datetime.datetime.utcnow()
        request = SentinelHubRequest(
            evalscript=ndwi_script,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(current_time - timedelta(hours=24), current_time),
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=self.lake_bbox,
            size=[512, 512],
            config=self.config
        )

        img = request.get_data()[0]
        if img.ndim == 3:
            img = img[:, :, 0]

        water_mask = (img > 0.2) & (~np.isnan(img))
        water_area = np.sum(water_mask) * (self.bbox_area / (512 * 512))

        if self.last_area and self.last_update:
            hours = (current_time - self.last_update).total_seconds() / 3600
            expansion_rate = (water_area - self.last_area) / (self.last_area * hours) * 100
        else:
            expansion_rate = 0.0

        self.last_area = water_area
        self.last_update = current_time

        return {
            "lake_area_km2": water_area,
            "expansion_rate_pct": expansion_rate,
            "image_timestamp": current_time.isoformat()
        }

    @retrying.retry(wait_exponential_multiplier=1000, stop_max_attempt_number=3)
    def get_weather_data(self) -> dict:
        try:
            client = openmeteo_requests.Client()
            params = {
                "latitude": self.lake_bbox.middle[1],
                "longitude": self.lake_bbox.middle[0],
                "hourly": ["temperature_2m", "precipitation", "snow_depth"],
                "forecast_days": 1,
                "timezone": "auto"
            }
            response = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
            hourly = response.Hourly()

            return {
                "temperature_c": hourly.Variables(0).Value(),
                "precipitation_mm": hourly.Variables(1).Value(),
                "snow_depth_m": hourly.Variables(2).Value(),
                "weather_timestamp": datetime.datetime.utcnow().isoformat()
            }
        except Exception as e:
            logging.error(f"Weather API error: {str(e)}")
            return {
                "temperature_c": None,
                "precipitation_mm": None,
                "snow_depth_m": None,
                "weather_timestamp": None
            }

    def fetch_all_data(self) -> pd.DataFrame:
        try:
            satellite = self.get_sentinel2_ndwi()
            weather = self.get_weather_data()

            combined = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                **(satellite or {}),
                **(weather or {})
            }
            return pd.DataFrame([combined])

        except Exception as e:
            logging.error(f"Data collection failed: {str(e)}")
            return pd.DataFrame()


def download_copernicus_dem(lat, lon, radius_km=1, save_path=None):
    try:
        buffer_deg = radius_km / 111
        min_lon = lon - buffer_deg
        min_lat = lat - buffer_deg
        max_lon = lon + buffer_deg
        max_lat = lat + buffer_deg

        logging.info(f"Querying Copernicus DEM for bounds: ({min_lon}, {min_lat}, {max_lon}, {max_lat})")

        # Use temporary directory for cloud environments if no save_path is provided
        if not save_path:
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, "dem_copernicus.tif")

        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=[min_lon, min_lat, max_lon, max_lat],
            limit=1,
        )

        items = list(search.items())
        if len(items) == 0:
            logging.error("No DEM found for this area.")
            return None

        item = items[0]
        signed_item = planetary_computer.sign(item)

        dem = stackstac.stack([signed_item], assets=["data"], bounds=(min_lon, min_lat, max_lon, max_lat), epsg=4326).squeeze()
        
        # Save the DEM to the specified path
        dem.rio.to_raster(save_path)

        logging.info(f"DEM downloaded and saved at {save_path}")

        return save_path

    except Exception as e:
        logging.error(f"Error downloading Copernicus DEM: {str(e)}")
        return None


def compute_dem_statistics(dem_path):
    try:
        with rasterio.open(dem_path) as src:
            elevation_data = src.read(1)
            transform = src.transform

        elevation_data = np.where(elevation_data == src.nodata, np.nan, elevation_data)
        if np.all(np.isnan(elevation_data)):
            logging.error("DEM contains only NaN values. Cannot compute statistics.")
            return None

        x_res = transform[0]
        y_res = -transform[4]

        grad_y, grad_x = np.gradient(elevation_data, y_res, x_res)
        slope_rad = np.arctan(np.sqrt(np.maximum(grad_x ** 2 + grad_y ** 2, 0)))
        slope_deg = np.degrees(slope_rad)

        zmin_m = np.nanmin(elevation_data)
        zmax_m = np.nanmax(elevation_data)
        zmean_m = np.nanmean(elevation_data)
        mean_slope_deg = np.nanmean(slope_deg)
        elevation_diff = zmax_m - zmin_m

        return {
            "mean_slope_deg": round(mean_slope_deg, 2),
            "elevation_diff_m": round(elevation_diff, 2),
            "zmin_m": round(zmin_m, 2),
            "zmax_m": round(zmax_m, 2),
            "zmean_m": round(zmean_m, 2)
        }
    except Exception as e:
        logging.error(f"Error computing DEM statistics: {str(e)}")
        return None


def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None


def prepare_features(lake_area_km2, dem_stats):
    features = pd.DataFrame(data=[[lake_area_km2,
                                   dem_stats["mean_slope_deg"],
                                   dem_stats["elevation_diff_m"],
                                   dem_stats["zmin_m"],
                                   dem_stats["zmax_m"],
                                   dem_stats["zmean_m"]]],
                            columns=[
                                'area_km2',
                                'slope_deg',
                                'elevation_diff',
                                'zmin_m',
                                'zmax_m',
                                'zmean_m'
                            ])
    return

def make_prediction(model, features):
    try:
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0].max()
        return {
            "risk_label": prediction,
            "confidence": round(proba * 100, 2)
        }
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {
            "risk_label": "Error",
            "confidence": 0.0
        }

