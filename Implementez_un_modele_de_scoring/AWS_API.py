from flask import Flask, request, jsonify
import numpy as np
import json
import pickle
import boto3
import tempfile

# Create an instance of the Flask class
app = Flask(__name__)

# AWS S3 configuration
s3 = boto3.client('s3')
bucket_name = 'elasticbeanstalk-eu-north-1-182399693743'

def download_from_s3(key):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        s3.download_fileobj(bucket_name, key, tmp_file)
        tmp_file.seek(0)
        return tmp_file.name


# Load the model
model_path = download_from_s3('model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)
    print("Model loaded successfully")

# Load preprocessing objects
imputer_path = download_from_s3('imputer.pkl')
with open(imputer_path, 'rb') as file:
    imputer = pickle.load(file)

scaler_path = download_from_s3('scaler.pkl')
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Load data from JSON file
data_path = download_from_s3('json_data.json')
with open(data_path, 'r') as f:
    data = json.load(f)

print("Type of data:", type(data))

# Convert JSON data to a dictionary for easy access
data_dict = {entry['SK_ID_CURR']: entry for entry in data}

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
        predictions = model.predict(client_features)

        # Interpret prediction (assuming binary classification: 0 = refused, 1 = accepted)
        decision = 'ACCEPTE' if predictions[0] == 1 else 'REFUSE'

        # Return plain text message
        return f"Prêt pour le client {client_id} : {decision}"

    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
