from flask import Flask, request, jsonify
import numpy as np
import json
import pickle
import boto3
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import gc

app = Flask(__name__)

s3 = boto3.client('s3')
bucket_name = 'elasticbeanstalk-eu-north-1-182399693743'

# Load the model and preprocessing objects
def load_pickle_from_s3(file_name):
    local_path = file_name.split('/')[-1]
    s3.download_file(bucket_name, file_name, local_path)
    with open(local_path, 'rb') as file:
        return pickle.load(file)

model = load_pickle_from_s3('model.pkl')
imputer = load_pickle_from_s3('imputer.pkl')
scaler = load_pickle_from_s3('scaler.pkl')

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

        return f"Decision for client {client_id}: {decision}"
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        gc.collect()  # Explicitly call garbage collection

def get_client_data(client_id):
    # Implement lazy loading or read from a smaller subset of your data
    # For example, read from a database or a smaller file
    pass

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
    app.run(host='0.0.0.0', port=5000)
