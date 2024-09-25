from flask import Flask, request, jsonify
import numpy as np
import pandas as pd  # Import pandas for JSON handling
import pickle
import boto3
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import gc
import logging

# Set up logging
logging.basicConfig(
    filename='app.log',  
    level=logging.DEBUG,  
    format='%(asctime)s %(levelname)s:%(message)s'
)

app = Flask(__name__)

# Example log message
logging.info("Flask application has started.")

s3 = boto3.client('s3')
bucket_name = 'elasticbeanstalk-eu-north-1-182399693743'

# Load the model and preprocessing objects from S3
def load_pickle_from_s3(file_name):
    local_path = file_name.split('/')[-1]
    s3.download_file(bucket_name, file_name, local_path)
    with open(local_path, 'rb') as file:
        return pickle.load(file)

model = load_pickle_from_s3('model.pkl')
imputer = load_pickle_from_s3('imputer.pkl')
scaler = load_pickle_from_s3('scaler.pkl')

# Load client data JSON from S3
def load_json_from_s3(file_name):
    local_path = file_name.split('/')[-1]
    try:
        s3.download_file(bucket_name, file_name, local_path)
        return pd.read_json(local_path)
    except Exception as e:
        logging.error(f"Failed to load JSON file from S3: {e}")
        raise  # Re-raise the exception after logging

# Load the client data 
client_data = load_json_from_s3('json_data.json')  # Load JSON data

# Function to retrieve client data based on ID
def get_client_data(client_id):
    logging.debug(f"Searching for client ID: {client_id}")  # Log the search ID
    
    # Ensure client_id is of the same type as SK_ID_CURR (int)
    client_row = client_data[client_data['SK_ID_CURR'] == int(client_id)]
    logging.debug(f"Client row found: {client_row}")  # Log the found row(s)
    
    if not client_row.empty:
        return client_row.iloc[0].to_dict()  # Convert row to dictionary
    return None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        request_data = request.get_json()
        client_id = request_data.get('SK_ID_CURR')

        if not client_id:
            return jsonify({"error": "Client ID is required"}), 400

        client_row = get_client_data(client_id)  # Lazy loading of client data
        if client_row is None:
            return jsonify({"error": "Client ID not found"}), 404
        
        # Prepare features for prediction
        client_features = np.array(list(client_row.values())).reshape(1, -1)

        # Preprocess the features
        client_features = imputer.transform(client_features)
        client_features = scaler.transform(client_features)

        predictions_proba = model.predict_proba(client_features)[:, 1]

        # Decision logic
        decision = classify_decision(predictions_proba)

        # Return plain text message
        return f"Decision for client {client_id}: {decision}"
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400
    finally:
        gc.collect()  # Explicitly call garbage collection

def classify_decision(predictions_proba):
    threshold_low = 0.4
    threshold_high = 0.6
    if threshold_low < predictions_proba < threshold_high:
        return 'MAYBE'
    elif predictions_proba > 0.5:
        return 'ACCEPTED'
    else:
        return 'REJECTED'

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
