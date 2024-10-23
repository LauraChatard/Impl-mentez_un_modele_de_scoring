from flask import Flask, request, jsonify
import numpy as np
import pandas as pd  # Import pandas for JSON handling
import pickle
import boto3
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import gc
import logging
import shap
from io import BytesIO

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
        df = pd.read_json(local_path)
        if df.empty:
            logging.warning(f"Loaded JSON file '{file_name}' is empty.")
        else:
            logging.info(f"Loaded JSON file '{file_name}' successfully with {df.shape[0]} records.")
        return df
    except Exception as e:
        logging.error(f"Failed to load JSON file from S3: {e}")
        raise

# Load the client data 
client_data = load_json_from_s3('reduced_data_dashboard.json')  # Load JSON data

def load_client_info(file_path, client_id):
    try:
        columns_of_interest = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'NAME_INCOME_TYPE', 'CODE_GENDER', 'NAME_CONTRACT_TYPE', 'CNT_CHILDREN']
        # Read only the necessary columns
        data = pd.read_csv(file_path, usecols=columns_of_interest)

        # Filter for the specific client ID
        client_row = data[data['SK_ID_CURR'] == client_id]
        if client_row.empty:
            logging.warning(f"Client ID {client_id} not found.")
            return None
        
        logging.info(f"Loaded client info for ID {client_id}")
        return client_row
    except Exception as e:
        logging.error(f"Failed to load client info: {e}")
        raise

file_name = 'filtered_application_train.csv'  # Nom du fichier dans le bucket S3

def load_data_from_s3(bucket, file_key):
    # Download the file from S3 and read it into a pandas DataFrame
    try:
        csv_obj = s3.get_object(Bucket=bucket, Key=file_key)
        body = csv_obj['Body']
        df = pd.read_csv(body)
        if df.empty:
            logging.warning(f"Loaded CSV file '{file_key}' is empty.")
        else:
            logging.info(f"Loaded CSV file '{file_key}' successfully with {df.shape[0]} records.")
        return df
    except Exception as e:
        logging.error(f"Failed to load CSV file from S3: {e}")
        raise


@app.route("/")
def home():
    return "API Flask Test"

def get_client_data(client_id):
    logging.debug(f"Searching for client ID: {client_id}")
    client_row = client_data[client_data['SK_ID_CURR'] == int(client_id)]
    logging.debug(f"Client row found: {client_row}")
    if not client_row.empty:
        return client_row.iloc[0].to_dict()
    logging.warning(f"Client ID {client_id} not found in data.")
    return None

def classify_decision(predictions_proba):
    threshold_low = 0.35
    threshold_high = 0.45
    if threshold_low <= predictions_proba <= threshold_high:
        return 'MAYBE'
    elif predictions_proba > 0.45:
        return 'ACCEPTED'
    else:
        return 'REJECTED'

@app.route("/predict", methods=["POST"])
def predict():
    try:
        request_data = request.get_json()
        client_id = request_data.get('SK_ID_CURR')

        if not client_id:
            return jsonify({"error": "Client ID is required"}), 400

        client_row_predict = get_client_data(client_id)
        if client_row_predict is None:
            return jsonify({"error": "Client ID not found"}), 404

        # Prepare features for prediction
        client_features = np.array([value for key, value in client_row_predict.items() if key not in ['TARGET', 'SK_ID_CURR']]).reshape(1, -1)
        feature_names = [key for key in client_row_predict.keys() if key not in ['TARGET', 'SK_ID_CURR']]
        logging.info(f"Shape of client features before imputation: {client_features.shape}")

       # Convert features to DataFrame with proper columns
        client_features_df = pd.DataFrame(client_features, columns=feature_names)

        # Preprocess the features
        client_features_imputed = imputer.transform(client_features_df)
        client_features_scaled = scaler.transform(client_features_imputed)
        logging.info(f"Shape of imputed client features: {client_features_imputed.shape}")
        logging.info(f"Shape of scaled client features: {client_features_scaled.shape}")

        predictions_proba = model.predict_proba(client_features_scaled)[:, 1]

        # Inside the predict function after scaling the features
        explainer = shap.TreeExplainer(model)

        # SHAP values will now return a list, and you need to extract the positive class
        shap_values = explainer.shap_values(client_features_scaled)

        # For binary classification, use the first class's SHAP values (for positive class)
        client_shap_values = shap_values[1][0]  # Adjust this based on the output structure

        # Create a DataFrame for interpretation
        shap_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": client_shap_values
        })

        # Now calculate the average SHAP values for accepted loans
        accepted_clients = client_data.sample(n=100, random_state=42)  # Randomly select 100 clients
        accepted_predictions_proba = model.predict_proba(scaler.transform(imputer.transform(accepted_clients[feature_names])))[:, 1]
        
        # Filter accepted clients
        accepted_clients = accepted_clients[accepted_predictions_proba > 0.45]

        if not accepted_clients.empty:
            # Calculate SHAP values for accepted clients
            accepted_shap_values = explainer.shap_values(scaler.transform(imputer.transform(accepted_clients[feature_names])))
            # Average SHAP values for accepted loans
            accepted_mean_shap_values = np.mean([accepted_shap_values[1][i] for i in range(len(accepted_clients))], axis=0)

            # Create a DataFrame for mean feature importance of accepted clients
            accepted_mean_shap_importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean SHAP Value": accepted_mean_shap_values
            })

            # Convert to dictionary for response
            accepted_mean_shap_importance_dict = accepted_mean_shap_importance_df.to_dict(orient='records')
        else:
            accepted_mean_shap_importance_dict = []

        # Calculate the average SHAP values for rejected loans
        rejected_clients = client_data.sample(n=100, random_state=42)
        rejected_predictions_proba = model.predict_proba(scaler.transform(imputer.transform(rejected_clients[feature_names])))[:, 1]
        
        # Filter accepted clients
        rejected_clients = rejected_clients[rejected_predictions_proba < 0.35]

        if not rejected_clients.empty:
            # Calculate SHAP values for accepted clients
            rejected_shap_values = explainer.shap_values(scaler.transform(imputer.transform(rejected_clients[feature_names])))
            # Average SHAP values for accepted loans
            rejected_mean_shap_values = np.mean([rejected_shap_values[1][i] for i in range(len(rejected_clients))], axis=0)

            # Create a DataFrame for mean feature importance of accepted clients
            rejected_mean_shap_importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean SHAP Value": rejected_mean_shap_values
            })

            # Convert to dictionary for response
            rejected_mean_shap_importance_dict = rejected_mean_shap_importance_df.to_dict(orient='records')
        else:
            rejected_mean_shap_importance_dict = []

        # Decision logic
        decision = classify_decision(predictions_proba)

        response_data = {
            "SK_ID_CURR": client_id,
            "decision": decision,
            "probability": float(predictions_proba),
            "shap_importances": shap_importance_df.to_dict(orient='records'),
            "accepted_mean_shap_importances": accepted_mean_shap_importance_dict,  # Add mean SHAP importances to the response
            "rejected_mean_shap_importances": rejected_mean_shap_importance_dict
        }

        return jsonify(response_data)


    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400
    finally:
        gc.collect()

@app.route("/client_info/<int:client_id>", methods=["GET"])
def client_info(client_id):
    # Charger les données clients depuis S3
    client_data = load_data_from_s3(bucket_name, file_name)

    # Filtrer pour obtenir uniquement les 100 premiers clients
    client_data = client_data.head(500)

    # Calculer les moyennes pour target = 0 et target = 1
    mean_income_target_0 = client_data[client_data['TARGET'] == 0]['AMT_INCOME_TOTAL'].mean()
    mean_income_target_1 = client_data[client_data['TARGET'] == 1]['AMT_INCOME_TOTAL'].mean()

    # Filtrer les données pour le client spécifié
    client_row = client_data[client_data['client_id'] == client_id]

    if client_row is not None:
        age = -(client_row.iloc[0]['DAYS_BIRTH'] / 365)  # Calculer l'âge
        result = {
            "AMT_INCOME_TOTAL": float(client_row.iloc[0]['AMT_INCOME_TOTAL']),
            "AMT_CREDIT": float(client_row.iloc[0]['AMT_CREDIT']),
            "age": int(age),
            "NAME_INCOME_TYPE": client_row.iloc[0]['NAME_INCOME_TYPE'],
            "CODE_GENDER": client_row.iloc[0]['CODE_GENDER'],
            "NAME_CONTRACT_TYPE": client_row.iloc[0]['NAME_CONTRACT_TYPE'],
            "CNT_CHILDREN": int(client_row.iloc[0]['CNT_CHILDREN']),
            "mean_income_target_0": float(mean_income_target_0),
            "mean_income_target_1": float(mean_income_target_1)
        }
        return jsonify(result)

    return jsonify({"error": "Client ID not found"}), 404
@app.route("/client_data", methods=["GET"])
def client_data_route():
    try:
        data_json = client_data.to_json(orient='records')
        return jsonify({"data": json.loads(data_json)})
    except Exception as e:
        logging.error(f"Error retrieving client data: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)