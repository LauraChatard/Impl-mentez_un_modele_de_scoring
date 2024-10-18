import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Function to get prediction from API
def get_prediction(sk_id_curr):
    url = "http://13.51.100.2:5000/predict"
    headers = {'Content-Type': 'application/json'}
    data = {'SK_ID_CURR': sk_id_curr}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to get client information from API
def get_client_info(client_id):
    url = f"http://13.51.100.2:5000/client_info/{client_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Error {response.status_code}: {response.text}"}

# Initialize session state variables if not already done
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'client_info' not in st.session_state:
    st.session_state.client_info = None
if 'show_graph' not in st.session_state:
    st.session_state.show_graph = False
if 'selected_feature' not in st.session_state:
    st.session_state.selected_feature = None
if 'top_features' not in st.session_state:
    st.session_state.top_features = []

# Streamlit interface
st.title("Loan Prediction Dashboard")

# Input for client ID
client_id = st.text_input("Enter Client ID")

# Define a color palette respecting WCAG standards
ACCESSIBLE_COLORS = {
    'rejected': '#8B0000',  # Dark Red
    'maybe': '#FFA500',     # Dark Orange
    'accepted': '#005B5C',  # Dark Teal
    'text': '#FFFFFF',      # White for text
    'object': '#FFFFFF'       # White for lines
}

# Initialize session state for feature importance display
if 'show_feature_importance' not in st.session_state:
    st.session_state.show_feature_importance = False

# Button to submit and get prediction
if st.button("Get Prediction"):
    if client_id:
        st.session_state.prediction_data = get_prediction(client_id)

        # Ajout d'un contrôle pour vérifier si la réponse est bien un dictionnaire
        if isinstance(st.session_state.prediction_data, dict):
            if st.session_state.prediction_data is None:
                st.error("No response from the API.")
            elif "error" in st.session_state.prediction_data:
                st.error(st.session_state.prediction_data["error"])
            else:
                decision = st.session_state.prediction_data['decision']
                prediction_prob = st.session_state.prediction_data['probability']  # Get the probability from the API response
                st.markdown(f"<h2 style='font-size: 30px;'>Decision: {decision}</h2>", unsafe_allow_html=True)
                st.session_state.show_graph = True
        else:
            # Affiche la réponse brute pour comprendre le problème
            st.error(f"Unexpected response format: {st.session_state.prediction_data}")

            # Define the colors and ranges for the segmented bar
            red_end = 0.35
            orange_start = red_end
            orange_end = 0.45
            blue_start = orange_end
            blue_end = 1.0

            # Create a Plotly figure
            fig = go.Figure()

            # Add a point for the client's probability
            fig.add_trace(go.Scatter(
                x=[prediction_prob],
                y=['Loan Acceptance Probability'],
                mode='markers+text',
                marker=dict(color=ACCESSIBLE_COLORS['object'], size=15, symbol='octagon-dot'),
                text=[f"Client {client_id}"],
                textposition='top right',
                hoverinfo='text',
                hovertext=[f"Probability: {prediction_prob:.2f}", f"Client {client_id}"],
                name=f'Client {client_id}',
                customdata=[client_id],
            ))

           # Add bars for each segment (rejected, maybe, accepted)
            fig.add_trace(go.Bar(
                y=['Loan Acceptance Probability'],
                x=[red_end],
                name='Rejected',
                orientation='h',
                marker=dict(color=ACCESSIBLE_COLORS['rejected']),
                hoverinfo='skip'  # Disable hover for bars
            ))
            fig.add_trace(go.Bar(
                y=['Loan Acceptance Probability'],
                x=[orange_end - orange_start],
                name='Maybe',
                orientation='h',
                marker=dict(color=ACCESSIBLE_COLORS['maybe']),
                base=orange_start,
                hoverinfo='skip'  # Disable hover for bars
            ))
            fig.add_trace(go.Bar(
                y=['Loan Acceptance Probability'],
                x=[blue_end - blue_start],
                name='Accepted',
                orientation='h',
                marker=dict(color=ACCESSIBLE_COLORS['accepted']),
                base=blue_start,
                hoverinfo='skip'  # Disable hover for bars
            ))


            # Set layout options
            fig.update_layout(
                barmode='stack',
                xaxis=dict(
                    range=[0, 1],
                    tickvals=[0, red_end, orange_end, blue_end],  # Valeurs de début et de fin
                    ticktext=['0', '0.35', '0.45', '1'],  # Texte des ticks
                    title='Probability',
                    showgrid=False
                ),
                yaxis=dict(
                    range=[-0.5, 0.5],  # Ajuster pour que la ligne corresponde bien
                    showticklabels=False
                ),
                title="Loan Acceptance Probability",
                showlegend=False,
                height=300,  # Adjust height for more space
                annotations=[
                    # Annotation for "Rejected" above the first bar
                    dict(
                        x=red_end / 2,  # Position in the middle of the "Rejected" bar
                        y=-0.5,  # Adjusted Y position above the bar
                        xref='x',
                        yref='y',
                        text="Rejected",  # Text to display
                        showarrow=False,
                        font=dict(size=12, color=ACCESSIBLE_COLORS['text'])  # Font color and size
                    ),
                    # Annotation for "Maybe" above the second bar
                    dict(
                        x=orange_start + (orange_end - orange_start) / 2,  # Position in the middle of the "Maybe" bar
                        y=-0.5,  # Adjusted Y position above the bar
                        xref='x',
                        yref='y',
                        text="Maybe",  # Text to display
                        showarrow=False,
                        font=dict(size=12, color=ACCESSIBLE_COLORS['text'])  # Font color and size
                    ),
                    # Annotation for "Accepted" above the third bar
                    dict(
                        x=blue_start + (blue_end - blue_start) / 2,  # Position in the middle of the "Accepted" bar
                        y=-0.5,  # Adjusted Y position above the bar
                        xref='x',
                        yref='y',
                        text="Accepted",  # Text to display
                        showarrow=False,
                        font=dict(size=12, color=ACCESSIBLE_COLORS['text'])  # Font color and size
                    )
                ]
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig)

# Show Feature Importance button only if the prediction is valid
if st.session_state.prediction_data and "error" not in st.session_state.prediction_data:
    if st.button("Show Feature Importance"):
        st.session_state.show_feature_importance = True


# Only show feature importance graph if the button is clicked
if st.session_state.show_feature_importance:
    st.markdown("### Top 5 Feature Importances")

    # Extract SHAP importances for the client
    shap_importances = st.session_state.prediction_data.get('shap_importances', [])

    if shap_importances:
        # Convert to DataFrame for easier handling
        shap_df = pd.DataFrame(shap_importances)
        
        # Select top 5 features based on absolute SHAP values
        top_5_features = shap_df.loc[shap_df['SHAP Value'].abs().nlargest(5).index]

        # Load SHAP importances and store top features in session state for average accepted loans
        response = requests.post("http://localhost:5004/predict", json={"SK_ID_CURR": client_id})
        if response.status_code == 200:
            data = response.json()
            avg_shap_importance_df = pd.DataFrame(data['accepted_mean_shap_importances'])
            avg_shap_importance_rejected_df = pd.DataFrame(data['rejected_mean_shap_importances'])

            # Prepare data for the initial combined bar chart
            combined_data_initial = pd.DataFrame({
                'Client SHAP Value': top_5_features.set_index('Feature')['SHAP Value'],
            })

            # Prepare data for combined bar chart
            combined_data = pd.DataFrame({
                'Client SHAP Value': top_5_features.set_index('Feature')['SHAP Value'],
                'Average Accepted SHAP Value': avg_shap_importance_df.set_index('Feature')['Mean SHAP Value'],
                'Average Rejected SHAP Value': avg_shap_importance_rejected_df.set_index('Feature')['Mean SHAP Value']
            }).abs()  # Convert to absolute values

            # Define colors based on the values of 'Client SHAP Value'
            colors = [
                ACCESSIBLE_COLORS['rejected'] if value < 0 else ACCESSIBLE_COLORS['accepted']
                for value in combined_data_initial['Client SHAP Value']
            ]

            # Plot initial combined bar chart for the top 5 features
            fig, ax = plt.subplots()
            combined_data_initial['Client SHAP Value'].plot(kind='barh', ax=ax, color=colors)
            ax.set_xlabel("SHAP Value")
            ax.set_title("Top 5 Feature Importances for Client")
            ax.invert_yaxis()  # To display the largest value on top
            st.pyplot(fig)

            # Dropdown to select feature for comparison
            feature_options = top_5_features['Feature'].tolist()
            selected_feature = st.selectbox("Select Feature to Compare", feature_options)

            # Extract selected feature values for comparison
            comparison_data = combined_data.loc[[selected_feature]]

            # Plot combined bar chart for the selected feature
            fig, ax = plt.subplots()
            comparison_data.plot(kind='barh', ax=ax, color=[ACCESSIBLE_COLORS['maybe'], ACCESSIBLE_COLORS['accepted'], ACCESSIBLE_COLORS['rejected']])
            ax.set_xlabel("SHAP Value (Absolute)")
            ax.set_title(f"Comparison of SHAP Values for {selected_feature}")
            ax.invert_yaxis()  # To display the largest value on top
            st.pyplot(fig)
        else:
            st.error(f"Error fetching mean SHAP importances: {response.status_code} - {response.text}")

    else:
        st.error("No SHAP importances available. Please check the prediction response.")

# Client Info collapsible section in sidebar
with st.sidebar.expander("Client Info"):
    if client_id:
        if st.session_state.client_info is None:
            st.session_state.client_info = get_client_info(client_id)
        
        if "error" not in st.session_state.client_info:
            st.sidebar.write(f"Age: {st.session_state.client_info['age']:.1f} years")
            st.sidebar.write(f"Gender: {st.session_state.client_info['CODE_GENDER']}")
            st.sidebar.write(f"Income Type: {st.session_state.client_info['NAME_INCOME_TYPE']}")
            st.sidebar.write(f"Loan Type: {st.session_state.client_info['NAME_CONTRACT_TYPE']}")
            st.sidebar.write(f"Children: {st.session_state.client_info['CNT_CHILDREN']}")
            st.sidebar.write(f"Income: {st.session_state.client_info['AMT_INCOME_TOTAL']} €")
        else:
            st.sidebar.error(st.session_state.client_info["error"])
    else:
        st.write("Please enter a Client ID and click 'Get Prediction'.")
