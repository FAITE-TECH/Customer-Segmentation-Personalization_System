# Import streamlit to build the interactive frontend
import streamlit as st
# Import requests to communicate with the FastAPI backend
import requests
# Import pandas for neat tabular presentation
import pandas as pd

# Define the base URL where the FastAPI backend is served
API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")

# Configure Streamlit page with a wide layout for dashboard-like UI
st.set_page_config(page_title="Customer Segmentation & Personalization", layout="wide")

# Create a title for the application
st.title("Customer Segmentation & Personalization")

# Build a sidebar to set backend URL and global controls
with st.sidebar:
    # Render a text input for the API endpoint to allow flexible setups
    api = st.text_input("Backend API URL", value=API_URL)
    # Persist the selected API URL for subsequent requests
    if api:
        API_URL = api
    # Provide a small info note to guide the user
    st.info("Ensure the FastAPI server is running before using the tools.")

# Define a convenience function to handle GET calls with error capture
def call_api(path: str, params: dict) -> dict:
    # Build the full URL for the request
    url = f"{API_URL}{path}"
    try:
        # Issue a GET request with the provided query parameters
        r = requests.get(url, params=params, timeout=30)
        # Raise an exception for non-2xx status codes
        r.raise_for_status()
        # Return the parsed JSON response
        return r.json()
    except Exception as e:
        # Render the error in the UI and return an empty dict
        st.error(f"API error: {e}")
        return {}

# Create tabs for each of the seven productized features
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "1) Customer Segmentation",
        "2) Personalized Campaigns",
        "3) Product Recommendations",
        "4) Dynamic Content",
        "5) Email Personalization",
        "6) Predictive Analytics",
        "7) Customer Journey Mapping",
    ]
)

# Build the segmentation tab UI
with tab1:
    # Provide input to capture a customer identifier
    cust_id = st.number_input("Customer ID", min_value=1, value=17850, step=1)
    # Provide a button to trigger the segmentation call
    if st.button("Get Segment", key="seg_btn"):
        # Call the backend segmentation endpoint
        data = call_api("/segment", {"customer_id": cust_id})
        # If response is populated, render the results
        if data:
            # Show the friendly segment name
            st.success(f"Segment: {data['segment_name']} (ID {data['segment_id']})")
            # Present RFM metrics in a table
            st.subheader("RFM Metrics")
            st.table(pd.DataFrame([data["rfm"]]))

# Build the personalized campaigns tab UI
with tab2:
    # Collect the customer identifier for campaign generation
    cust_id = st.number_input("Customer ID ", min_value=1, value=17850, step=1, key="cmp_cust")
    # Provide a button to generate a campaign
    if st.button("Generate Campaign Plan", key="cmp_btn"):
        # Invoke the campaign endpoint
        data = call_api("/personalized_campaigns", {"customer_id": cust_id})
        # Render the plan if available
        if data:
            # Show top-line campaign details
            st.success(f"Channel: {data['recommended_channel']} | Offer: {data['proposed_offer']} | Tone: {data['creative_tone']}")
            # Display example creative assets
            st.write("**Headline Suggestion**")
            st.write(data["example_headline"])
            st.write("**Copy Suggestion**")
            st.write(data["example_copy"])

# Build the recommendation tab UI
with tab3:
    # Accept a customer id to inform segment-aware fallback
    cust_id = st.number_input("Customer ID  ", min_value=1, value=17850, step=1, key="rec_cust")
    # Accept optional recent items to seed item-to-item recommendations
    recent = st.text_input("Recent Items (comma-separated StockCodes)", value="85123A, 71053")
    # Accept the number of recommendations desired
    k = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)
    # Trigger recommendation generation on button click
    if st.button("Recommend Products", key="rec_btn"):
        # Call the recommendations endpoint with both signals
        data = call_api("/recommendations", {"customer_id": cust_id, "recent_items": recent, "top_k": k})
        # Render structured results if present
        if data and data.get("recommendations"):
            # Display recommendations in a table for clarity
            st.subheader("Recommended Products")
            st.table(pd.DataFrame(data["recommendations"]))

# Build the dynamic content tab UI
with tab4:
    # Request which customer id to personalize the homepage for
    cust_id = st.number_input("Customer ID   ", min_value=1, value=17850, step=1, key="dyn_cust")
    # Generate dynamic content on demand
    if st.button("Generate Homepage Content", key="dyn_btn"):
        # Invoke the backend dynamic content endpoint
        data = call_api("/dynamic_content", {"customer_id": cust_id})
        # Render if we got a valid response
        if data:
            # Show the banner text that should appear above the fold
            st.success(f"Banner: {data['homepage_banner']}")
            # Show the content strategy for reference
            st.info(f"Strategy: {data['rendering_strategy']}")
            # List the featured products to slot into hero modules
            st.subheader("Featured Products")
            st.table(pd.DataFrame(data["featured_products"]))

# Build the email personalization tab UI
with tab5:
    # Let the user target a customer id for the email
    cust_id = st.number_input("Customer ID    ", min_value=1, value=17850, step=1, key="email_cust")
    # Produce email content when requested
    if st.button("Create Personalized Email", key="email_btn"):
        # Hit the email personalization endpoint
        data = call_api("/email_personalization", {"customer_id": cust_id})
        # If payload exists display the email subject and body
        if data:
            # Subject line visualization
            st.subheader("Subject")
            st.code(data["subject"])
            # Body visualization
            st.subheader("Body")
            st.text_area("Email Body", value=data["body"], height=220)

# Build the predictive analytics tab UI
with tab6:
    # Select a customer to score for repeat purchase probability
    cust_id = st.number_input("Customer ID     ", min_value=1, value=17850, step=1, key="pred_cust")
    # Run the prediction by clicking the button
    if st.button("Score Repeat Purchase Likelihood", key="pred_btn"):
        # Call the predictive analytics endpoint
        data = call_api("/predictive_analytics", {"customer_id": cust_id})
        # Render probability and action suggestion
        if data:
            # Display the computed probability formatted as a percentage
            st.metric("Repeat Purchase Probability", f"{data['repeat_purchase_probability']*100:.1f}%")
            # Display the recommended action to take next
            st.success(f"Recommended Action: {data['recommended_action']}")

# Build the customer journey mapping tab UI
with tab7:
    # Choose which customer to display a journey for
    cust_id = st.number_input("Customer ID      ", min_value=1, value=17850, step=1, key="journey_cust")
    # Trigger the journey retrieval
    if st.button("Show Journey", key="journey_btn"):
        # Call the backend journey endpoint
        data = call_api("/journey_mapping", {"customer_id": cust_id})
        # Render events in chronological order
        if data and data.get("events"):
            # Create a dataframe from the list of dictionaries
            df = pd.DataFrame(data["events"])
            # Sort by timestamp for coherent timeline
            df = df.sort_values("timestamp")
            # Display the timeline in a table for inspection
            st.subheader("Journey Timeline")
            st.dataframe(df, use_container_width=True)
            # Display the next best action recommendation
            st.success(f"Next Best Action: {data['next_best_action']}")
