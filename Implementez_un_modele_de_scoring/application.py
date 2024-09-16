from flask import Flask, request, jsonify
import mlflow
import numpy as np
import json
import pickle
import boto3
import os

app = Flask(__name__)

# AWS S3 configuration
s3 = boto3.client('s3')
bucket_name = 'elasticbeanstalk-eu-north-1-182399693743'

# Function to download a file from S3
def download_from_s3(file_key, download_path):
    s3.download_file(bucket_name, file_key, download_path)

# Download model.pkl, imputer.pkl, and scaler.pkl from S3
download_from_s3('model.pkl', 'model.pkl')
download_from_s3('imputer.pkl', 'imputer.pkl')
download_from_s3('scaler.pkl', 'scaler.pkl')
download_from_s3('feature_names.json', 'feature_names.json')

# Load feature names from file
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    print("Model loaded successfully")

# Load the imputer and scaler
with open('imputer.pkl', 'rb') as file:
    imputer = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load data from JSON file (you can add a similar download function for this if needed)
data_path = 'json_data.json'
with open(data_path, 'r') as f:
    data = json.load(f)

print("Type of data:", type(data))

# Convert JSON data to a dictionary for easy access
data_dict = {entry['SK_ID_CURR']: entry for entry in data}

@app.route("/")
def home():
    return "API Flask Test"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        request_data = request.get_json()
        client_id = request_data.get('SK_ID_CURR')

        if not client_id:
            return jsonify({"error": "Client ID is required"}), 400

        # Extract the client row from the data
        client_row = data_dict.get(client_id)
        if client_row is None:
            return jsonify({"error": "Client ID not found"}), 404

        # Drop the target column and prepare features for prediction
        client_row = {key: value for key, value in client_row.items() if key in feature_names}
        features_data = np.array([client_row[feature] for feature in feature_names]).reshape(1, -1)

        # Apply the preprocessing steps: imputation and scaling
        features_data = imputer.transform(features_data)
        features_data = scaler.transform(features_data)

        # Make predictions
        predictions = model.predict(features_data)

        # Return the formatted response
        decision = "accepté" if predictions[0] == 1 else "refusé"
        return f"Prêt pour le client **{client_id}** : {decision}"
    
    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5003)
