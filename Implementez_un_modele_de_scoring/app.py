from flask import Flask, request, jsonify
import numpy as np
import json
import pickle
import boto3
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Initialize a session using your AWS credentials
s3 = boto3.client('s3')

# Specify the S3 bucket name
bucket_name = 'elasticbeanstalk-eu-north-1-182399693743'

# Function to download and load a pickle file
def load_pickle_from_s3(file_name):
    # Download the file to a local path
    local_path = file_name.split('/')[-1]  # Get the local file name from the S3 path
    s3.download_file(bucket_name, file_name, local_path)

    # Load the model
    with open(local_path, 'rb') as file:
        return pickle.load(file)

# Load the model
model = load_pickle_from_s3('model.pkl')
print("Model loaded successfully")

# Load preprocessing objects
imputer = load_pickle_from_s3('imputer.pkl')
print("Imputer loaded successfully")

scaler = load_pickle_from_s3('scaler.pkl')
print("Scaler loaded successfully")

# Load data from JSON file
data_path = 'json_data.json'  # Local path for the JSON file
s3.download_file(bucket_name, data_path, data_path)
print("Data download successfully")

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

        # Ensure client_id is an integer
        client_id = int(client_id) if isinstance(client_id, str) else client_id

        print("Received client ID:", client_id)

        # Extract the client row from the data
        client_row = data_dict.get(client_id)
        print("Client row:", client_row)

        if client_row is None:
            return jsonify({"error": "Client ID not found"}), 404
        
        # Remove the target column if it exists in the client row
        target_column = 'TARGET' 
        client_row.pop(target_column, None)  # Safely remove the target column if it exists
        
        # Prepare features for prediction by extracting only the feature names
        client_features = np.array(list(client_row.values())).reshape(1, -1)

        # Preprocess the features
        client_features = imputer.transform(client_features)
        client_features = scaler.transform(client_features)

        print("Preprocessed features data for prediction:", client_features)

        # Make predictions
        predictions_proba = model.predict_proba(client_features)[:, 1]  # Probability for the positive class

        # Define a threshold for the "maybe" class
        threshold_low = 0.4
        threshold_high = 0.6
        if threshold_low < predictions_proba < threshold_high:
            decision = 'MAYBE'
        elif predictions_proba > 0.5:
            decision = 'ACCEPTED'
        else:
            decision = 'REJECTED'

        # Return plain text message
        return f"Decision for client {client_id}: {decision}"

    
    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
