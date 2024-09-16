from flask import Flask, request, jsonify
import mlflow
import numpy as np
import json

app = Flask(__name__)

# Correct model path: Point to the directory containing the model
model_path = "/Users/laurachatard/Documents/Vie Etudiante/Openclassrooms/projet_7/mlruns/471647679769818948/52dca41b1e9f4d36a55ecb92b8955726/artifacts/best_model"
model = mlflow.sklearn.load_model(model_path)  # Load the model using MLflow

@app.route("/")
def home():
    return "API Flask Test"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request
        features = np.array(data["features"])  # Convert data to numpy array (assumed format)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5003)
