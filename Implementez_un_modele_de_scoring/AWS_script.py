import streamlit as st
import requests

# Function to get prediction from API
def get_prediction(sk_id_curr):
    url = "https://scoring-client.eu-north-1.elasticbeanstalk.com /api/predict"
    headers = {'Content-Type': 'application/json'}
    data = {'SK_ID_CURR': sk_id_curr}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        return response.text  # Returning plain text response
    else:
        return f"Error: {response.status_code}, {response.text}"

# Streamlit interface
st.title("Loan Prediction")

# Input for client ID
client_id = st.text_input("Enter Client ID")

# Button to submit and get prediction
if st.button("Get Prediction"):
    if client_id:
        prediction = get_prediction(client_id)
        # Displaying the prediction with client ID in bold using Markdown
        st.markdown(f"Prêt pour le client **{client_id}** : {prediction.split(': ')[1]}")
    else:
        st.write("Please enter a Client ID.")
