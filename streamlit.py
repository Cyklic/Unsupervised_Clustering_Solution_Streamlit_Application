import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import logging
import traceback
from src.visualization.visualize import Cluster
from src.data.make_dataset import load_data

# Configure logging
logging.basicConfig(
    filename="mall_segmentation_app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

logging.info("Mall Customer Segmentation app started.")

# Set the page title and description
st.markdown("<h1 style='text-align: center;'>Mall Customer Segmentation Model</h1>", unsafe_allow_html=True)
st.write("""
This app divides mall customers into various groups based on their age, annual income, and spending score.
""")

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()

# Load dataset
try:
    df = load_data("data/raw/mall_customers.csv")
    logging.info("Mall customer data loaded successfully.")
except Exception as e:
    logging.error("Error loading mall customer dataset.")
    logging.error(traceback.format_exc())
    st.error("Error loading customer data. Please try again later.")
    st.stop()

# Load the pre-trained model
try:
    with open("models/Kmodel.pkl", "rb") as k_pickle:
        k_model = pickle.load(k_pickle)
    logging.info("K-means model loaded successfully.")
except Exception as e:
    logging.error("Failed to load K-means model.")
    logging.error(traceback.format_exc())
    st.error("Error loading model. Please check the file and try again.")
    st.stop()

# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.markdown("<h3 style='text-align: center;'>Mall Customer Details</h3>", unsafe_allow_html=True)
    
    Age = st.number_input("Age", min_value=0, step=10)
    Annual_Income = st.number_input("Annual Income in Thousands", min_value=0, step=10)
    Spending_Score = st.number_input("Spending Score", min_value=0, step=10)
    
    submitted = st.form_submit_button("Group Mall Customers")

# Handle prediction and display
if submitted:
    try:
        # Prepare the input for prediction. This has to go in the same order as it was trained
        prediction_input = [[Age, Annual_Income, Spending_Score]]
        
        # Make prediction
        new_prediction = k_model.predict(prediction_input)[0]

        # Display result
        st.subheader("Prediction Result:")
        st.success(f"Your Mall Customer grouping is: {new_prediction}")
        logging.info(f"Prediction successful: Cluster {new_prediction}")

        # Put this data back in to the main dataframe corresponding to each observation
        df['Cluster'] = k_model.labels_
        try:
            Cluster(df, Annual_Income, Spending_Score)
            st.image("Clusters.png")
            st.image("Silhouette_Score.png")
            logging.info("Cluster visualization displayed.")
        except Exception as e:
            logging.warning("Failed to generate/display cluster images.")
            logging.warning(traceback.format_exc())
            st.warning("Could not display cluster images.")

    except Exception as e:
        logging.error("Error during prediction or clustering.")
        logging.error(traceback.format_exc())
        st.error("An error occurred during grouping. Please check your inputs and try again.")

st.write(
    """We used a machine learning (K-means clustering) model to group the mall customers."""
)
