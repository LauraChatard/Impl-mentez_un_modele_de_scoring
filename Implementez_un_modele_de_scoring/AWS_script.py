import streamlit as st
import requests
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

# Function to get prediction from API
def get_prediction(sk_id_curr):
    url = "http://13.51.100.2:5000/predict"
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
        # Display the prediction with client ID in bold using Markdown
        st.markdown(prediction)  # Directly display the plain text response
    else:
        st.write("Please enter a Client ID.")
