# combined_app.py
import streamlit as st
import pandas as pd

# import your pages
from pages.trends import show_trends_page
from pages.classification import show_classification_page
from pages.clustering import show_clustering_page

st.set_page_config(page_title="EV Analytics Platform", layout="wide")
st.title("⚡ Unified EV Analytics Platform")

@st.cache_data
def load_data(path="Electric_Vehicle_Population_Data.csv"):
    return pd.read_csv(path)

df = load_data()

# Sidebar navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", [
    "EV Trends & Forecasting",
    "CAFV Eligibility Classification",
    "EV Clustering & Infrastructure"
])

if page == "EV Trends & Forecasting":
    show_trends_page(df)
elif page == "CAFV Eligibility Classification":
    show_classification_page(df)
elif page == "EV Clustering & Infrastructure":
    show_clustering_page(df)
