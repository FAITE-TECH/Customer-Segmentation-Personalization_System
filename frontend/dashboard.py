import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Customer Segmentation & Personalization", layout="wide")

st.title("🛒 Customer Segmentation & Personalization Dashboard")

tabs = st.tabs([
    "Customer Segmentation",
    "Personalized Marketing",
    "Product Recommendations",
    "Dynamic Content",
    "Email Personalization",
    "Predictive Analytics",
    "Customer Journey Mapping"
])

# 1. Customer Segmentation
with tabs[0]:
    customer_id = st.number_input("Enter Customer ID (optional)", min_value=0, value=0)
    if st.button("Segment Customer"):
        payload = {"customer_id": int(customer_id) if customer_id > 0 else None}
        response = requests.post(f"{API_URL}/segment_customers", json=payload)
        st.json(response.json())

# 2. Personalized Marketing
with tabs[1]:
    segment = st.selectbox("Select Segment", ["Low", "Mid", "High", "VIP"])
    if st.button("Get Campaign"):
        payload = {"customer_segment": segment}
        response = requests.post(f"{API_URL}/personalized_marketing", json=payload)
        st.json(response.json())

# 3. Product Recommendations
with tabs[2]:
    cid = st.number_input("Customer ID", min_value=1)
    if st.button("Get Recommendations"):
        payload = {"customer_id": cid}
        response = requests.post(f"{API_URL}/product_recommendations", json=payload)
        st.json(response.json())

# 4. Dynamic Content
with tabs[3]:
    cid = st.number_input("Customer ID for Dynamic Content", min_value=1)
    if st.button("Get Personalized Content"):
        payload = {"customer_id": cid}
        response = requests.post(f"{API_URL}/dynamic_content", json=payload)
        st.json(response.json())

# 5. Email Personalization
with tabs[4]:
    cid = st.number_input("Customer ID for Email", min_value=1)
    product_name = st.text_input("Product Name")
    if st.button("Generate Email"):
        payload = {"customer_id": cid, "product_name": product_name} # sample productname . "Popular Item A"
        response = requests.post(f"{API_URL}/email_personalization", json=payload)
        st.json(response.json())

# 6. Predictive Analytics
with tabs[5]:
    cid = st.number_input("Customer ID for Prediction", min_value=1)
    if st.button("Predict Behavior"):
        payload = {"customer_id": cid}
        response = requests.post(f"{API_URL}/predictive_analytics", json=payload)
        
        st.json(response.json())

# 7. Customer Journey Mapping
with tabs[6]:
    cid = st.number_input("Customer ID for Journey", min_value=1)
    if st.button("Map Journey"):
        payload = {"customer_id": cid}
        response = requests.post(f"{API_URL}/journey_mapping", json=payload)
        st.json(response.json())
