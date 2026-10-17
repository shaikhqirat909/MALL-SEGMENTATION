
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Mall Customer Segmentation")
st.write("Enter customer details to predict the customer cluster.")

# Load trained model
with open("kmeans_model.pkl", "rb") as file:
    kmeans = pickle.load(file)

# Customer inputs
income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0,
    value=50.0,
    step=1.0
)

spending_score = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50,
    step=1
)

if st.button("Predict Cluster"):
    customer = pd.DataFrame(
        [[income, spending_score]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    cluster = kmeans.predict(customer)[0]

    st.success(f"Customer belongs to Cluster {cluster}")

    # Show customer on graph
    fig, ax = plt.subplots()

    ax.scatter(
        income,
        spending_score,
        s=150,
        marker="X",
        label="Your Customer"
    )

    ax.set_title("Customer Segmentation Graph")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.legend()

    st.pyplot(fig)
