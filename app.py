from flask import Flask, jsonify, request
import backend  # Assuming the backend.py file is in the same directory

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the GLOF Monitoring API!"

@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    try:
        # Sample lake coordinates (you can get these from the request)
        lat = float(request.args.get('lat', 53.5587))  # default to 53.5587
        lon = float(request.args.get('lon', 108.1650))  # default to 108.1650
        radius_km = int(request.args.get('radius_km', 5))  # default to 5 km radius

        # Initialize the RealTimeGLOFMonitor with provided coordinates
        monitor = backend.RealTimeGLOFMonitor((lat, lon), radius_km)

        # Fetch all data
        data = monitor.fetch_all_data()

        if not data.empty:
            return jsonify(data.to_dict(orient='records')[0]), 200  # Return data as JSON
        else:
            return jsonify({"error": "Data collection failed"}), 500

    except Exception as e:
        return jsonify({"error": f"System error: {str(e)}"}), 500

@app.route('/get_dem_statistics', methods=['GET'])
def get_dem_statistics():
    try:
        lat = float(request.args.get('lat', 53.5587))
        lon = float(request.args.get('lon', 108.1650))
        radius_km = int(request.args.get('radius_km', 1))

        # Download Copernicus DEM
        dem_file = backend.download_copernicus_dem(lat, lon, radius_km)
        
        if dem_file:
            # Compute DEM statistics
            stats = backend.compute_dem_statistics(dem_file)
            if stats:
                return jsonify(stats), 200
            else:
                return jsonify({"error": "DEM statistics could not be computed"}), 500
        else:
            return jsonify({"error": "DEM file not found"}), 500

    except Exception as e:
        return jsonify({"error": f"System error: {str(e)}"}), 500

@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    try:
        # Get data from the request (JSON body)
        data = request.get_json()

        if not data or 'lake_area_km2' not in data or 'dem_stats' not in data:
            return jsonify({"error": "Invalid input data"}), 400

        lake_area_km2 = data['lake_area_km2']
        dem_stats = data['dem_stats']

        # Prepare features for prediction
        features = backend.prepare_features(lake_area_km2, dem_stats)

        # Load the model and make prediction
        model = backend.load_model('best_rf_model.pkl')
        prediction = backend.make_prediction(model, features)

        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": f"System error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
