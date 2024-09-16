from flask import Flask, request, jsonify
import mlflow
import numpy as np
import json
import boto3
import os

app = Flask(__name__)

# AWS S3 Configuration
s3_client = boto3.client('s3')
bucket_name = 'elasticbeanstalk-eu-north-1-182399693743'
model_key = 'model.pkl'
feature_names_key = 'feature_names.json'
local_model_path = '/tmp/model.pkl'  # Temporary path to store the downloaded model
local_feature_names_path = '/tmp/feature_names.json'  # Temporary path to store the downloaded feature names

# Download model from S3
def download_from_s3(key, local_path):
    s3_client.download_file(bucket_name, key, local_path)

# Download model and feature names
download_from_s3(model_key, local_model_path)
download_from_s3(feature_names_key, local_feature_names_path)

# Load the model
model = mlflow.sklearn.load_model(local_model_path)

# Load feature names
with open(local_feature_names_path, 'r') as f:
    feature_names = json.load(f)

@app.route("/")
def home():
    return "API Flask Test"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request
        print(data)
        # Print incoming data for debugging
        print("Incoming data:", data)

        # Extract only the features needed for prediction
        features_data = {key: data[key] for key in feature_names if key in data}
        
        # Print the features data being used
        print("Features data:", features_data)

        if len(features_data) != len(feature_names):
            # Print which features are missing
            missing_features = set(feature_names) - set(features_data.keys())
            print("Missing features:", missing_features)
            return jsonify({"error": "Incomplete feature data"}), 400

        # Prepare features for prediction
        features = np.array([features_data[feature] for feature in feature_names]).reshape(1, -1)

        # Make predictions
        predictions = model.predict(features)
        
        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        # Print the exception for debugging
        print("Exception occurred:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5003)
