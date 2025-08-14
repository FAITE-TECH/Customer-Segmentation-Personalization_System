# Import FastAPI to build the web API
from fastapi import FastAPI, HTTPException
# Import CORS middleware to allow the frontend to call the backend
from fastapi.middleware.cors import CORSMiddleware
# Import data structures for typing clarity
from typing import List, Optional, Dict, Any
# Import joblib to load trained artifacts from disk
import joblib
# Import pandas for lightweight data handling in endpoints
import pandas as pd
# Import pathlib to manage filesystem-safe paths
from pathlib import Path

# Create a FastAPI application instance
app = FastAPI(title="Customer Segmentation & Personalization API", version="1.0.0")

# Configure CORS to permit local Streamlit development and any additional origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace wildcard with exact frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a dictionary to hold loaded artifacts for reuse across requests
ARTIFACTS: Dict[str, Any] = {}

# Define a helper function to safely read small lookup tables saved during training
def _read_pickle(name: str) -> Any:
    # Compute absolute path to the models directory
    model_dir = Path(__file__).parent / "models"
    # Build full path for the target artifact
    path = model_dir / name
    # Validate that the artifact exists
    if not path.exists():
        # Raise HTTP error if the artifact is missing
        raise HTTPException(status_code=500, detail=f"Model artifact not found: {path}")
    # Load and return the artifact via joblib
    return joblib.load(path)

# Define the startup event to load artifacts once when the server boots
@app.on_event("startup")
def load_models_on_startup() -> None:
    # Load the scaler for RFM features
    ARTIFACTS["rfm_scaler"] = _read_pickle("rfm_scaler.pkl")
    # Load the KMeans customer segmentation model
    ARTIFACTS["kmeans"] = _read_pickle("kmeans_segmentation.pkl")
    # Load the logistic regression classifier for repeat purchase prediction
    ARTIFACTS["repeat_clf"] = _read_pickle("repeat_purchase_clf.pkl")
    # Load the list of RFM feature names used by the models
    ARTIFACTS["rfm_features"] = _read_pickle("rfm_features.pkl")
    # Load the customer RFM table to serve stats quickly
    ARTIFACTS["rfm_table"] = _read_pickle("rfm_table.pkl")
    # Load product popularity per segment for segment-aware recommendations
    ARTIFACTS["segment_product_rank"] = _read_pickle("segment_product_rank.pkl")
    # Load a simple co-occurrence based recommender lookup
    ARTIFACTS["cooc_recommend"] = _read_pickle("cooc_recommend.pkl")
    # Load product metadata dictionary for human-friendly labels
    ARTIFACTS["product_meta"] = _read_pickle("product_meta.pkl")
    # Load segment labels for user-facing names
    ARTIFACTS["segment_labels"] = _read_pickle("segment_labels.pkl")
    # Load customer journeys constructed from invoices
    ARTIFACTS["journeys"] = _read_pickle("journeys.pkl")

# Define a helper to get a single customer's RFM row or raise a 404
def _get_customer_rfm(customer_id: int) -> pd.Series:
    # Access the precomputed RFM table
    rfm = ARTIFACTS["rfm_table"]
    # Check for membership of the requested customer identifier
    if customer_id not in rfm.index:
        # Raise a not found error if the customer is missing
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    # Return the RFM row as a pandas Series
    return rfm.loc[customer_id]

# Define the root endpoint to provide a quick health and info message
@app.get("/")
def root() -> Dict[str, Any]:
    # Return a static message confirming service availability
    return {"message": "Customer Segmentation & Personalization API is running"}

# Define the segmentation endpoint to compute and return a customer's segment
@app.get("/segment")
def segment(customer_id: int) -> Dict[str, Any]:
    # Retrieve the raw RFM values for the customer
    row = _get_customer_rfm(customer_id)
    # Extract the numerical RFM feature values in the expected order
    X = row[ARTIFACTS["rfm_features"]].values.reshape(1, -1)
    # Transform features using the same scaler from training
    Xs = ARTIFACTS["rfm_scaler"].transform(X)
    # Predict the cluster number with the trained KMeans model
    cluster = int(ARTIFACTS["kmeans"].predict(Xs)[0])
    # Resolve the friendly segment label for better readability
    segment_name = ARTIFACTS["segment_labels"].get(cluster, f"Segment {cluster}")
    # Return the RFM metrics and the assigned segment for UI display
    return {
        "customer_id": customer_id,
        "rfm": row[ARTIFACTS["rfm_features"]].to_dict(),
        "segment_id": cluster,
        "segment_name": segment_name,
    }

# Define the personalized campaigns endpoint to generate a tailored offer
@app.get("/personalized_campaigns")
def personalized_campaigns(customer_id: int) -> Dict[str, Any]:
    # Obtain segmentation details to drive campaign logic
    seg = segment(customer_id)
    # Extract the segment identifier for rules
    seg_id = seg["segment_id"]
    # Define simple rule-based campaign templates keyed by segment identifier
    rules = {
        0: {"channel": "Email", "offer": "10% off new arrivals", "tone": "Value-driven"},
        1: {"channel": "SMS", "offer": "Loyalty bonus: 2x points", "tone": "Exclusive"},
        2: {"channel": "Push", "offer": "Limited-time bundle deal", "tone": "Urgent"},
        3: {"channel": "Email", "offer": "Free shipping over $30", "tone": "Friendly"},
    }
    # Select the rule set or default to a generic campaign
    plan = rules.get(seg_id, {"channel": "Email", "offer": "Welcome 5% off", "tone": "Warm"})
    # Build a creative suggestion payload for the frontend to render
    return {
        "customer_id": customer_id,
        "segment_name": seg["segment_name"],
        "recommended_channel": plan["channel"],
        "proposed_offer": plan["offer"],
        "creative_tone": plan["tone"],
        "example_headline": f"{seg['segment_name']}: {plan['offer']}",
        "example_copy": f"As a valued {seg['segment_name']} member, enjoy {plan['offer']} today.",
    }

# Define a function to convert product codes to readable names for UI
def _labels_for_products(stockcodes: List[str]) -> List[Dict[str, str]]:
    # Access the product metadata dictionary for names
    meta = ARTIFACTS["product_meta"]
    # Build a list of dictionaries pairing code to description
    return [{"stockcode": c, "description": meta.get(c, f"Item {c}")} for c in stockcodes]

# Define the recommendations endpoint to suggest products for a customer
@app.get("/recommendations")
def recommendations(customer_id: Optional[int] = None, recent_items: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
    # Initialize a list to collect recommended product identifiers
    recs: List[str] = []
    # If recent_items are provided, do co-occurrence based next-best-item suggestions
    if recent_items:
        # Split comma-separated stock codes and normalize spaces
        items = [x.strip() for x in recent_items.split(",") if x.strip()]
        # For each item, extend recommendations using learned co-occurrence mapping
        for it in items:
            # Get candidates for this item and append while preserving order
            recs.extend(ARTIFACTS["cooc_recommend"].get(it, []))
    # If customer id is provided and we still need more recommendations, use segment popularity
    if customer_id is not None and len(recs) < top_k:
        # Resolve the customer segment to pull segment-aware top items
        seg_info = segment(customer_id)
        # Read the ranked list for the target segment
        seg_rank = ARTIFACTS["segment_product_rank"].get(seg_info["segment_id"], [])
        # Extend the recommendation list with segment-popular products
        recs.extend(seg_rank)
    # De-duplicate while preserving original ranking
    seen = set()
    # Build the final ranked list without repeats
    unique = [x for x in recs if not (x in seen or seen.add(x))]
    # Truncate the list to the requested size
    final = unique[:top_k]
    # Convert stock codes to labeled objects for UI
    labeled = _labels_for_products(final)
    # Return the final recommendation payload
    return {"customer_id": customer_id, "input_items": recent_items, "recommendations": labeled}

# Define the dynamic content endpoint to generate homepage blocks for a customer
@app.get("/dynamic_content")
def dynamic_content(customer_id: int) -> Dict[str, Any]:
    # Fetch the customer segment information first
    seg = segment(customer_id)
    # Map each segment to a dynamic banner and featured product strategy
    content_map = {
        "High-Value Loyalists": {
            "banner": "Welcome back! Double points on your favorites today.",
            "strategy": "Show premium bestsellers",
        },
        "Price-Sensitive Savers": {
            "banner": "Hot deals picked for you. Save more on bundles!",
            "strategy": "Show discounted sets",
        },
        "Seasonal Buyers": {
            "banner": "Fresh seasonal picks just arrived. Limited stock!",
            "strategy": "Show seasonal arrivals",
        },
        "New or Lapsed": {
            "banner": "Nice to see you! Enjoy free shipping on first order.",
            "strategy": "Show entry-level bestsellers",
        },
    }
    # Retrieve the content plan or fall back to a safe default
    plan = content_map.get(seg["segment_name"], {"banner": "Welcome! Explore popular picks.", "strategy": "Show overall top"})
    # Recommend a short list of items using segment rank as featured products
    featured_codes = ARTIFACTS["segment_product_rank"].get(seg["segment_id"], [])[:4]
    # Convert the codes to labeled UI-friendly items
    featured = _labels_for_products(featured_codes)
    # Return the assembled dynamic content response
    return {
        "customer_id": customer_id,
        "segment_name": seg["segment_name"],
        "homepage_banner": plan["banner"],
        "rendering_strategy": plan["strategy"],
        "featured_products": featured,
    }

# Define the email personalization endpoint to produce subject and body text
@app.get("/email_personalization")
def email_personalization(customer_id: int) -> Dict[str, Any]:
    # Derive the segment and campaign for this customer
    seg = segment(customer_id)
    # Build the personalized campaign suggestion
    camp = personalized_campaigns(customer_id)
    # Create a dynamic subject line tying the segment and offer
    subject = f"{seg['segment_name']}: {camp['proposed_offer']} just for you"
    # Create a simple, readable email body referencing recommendations
    rec = recommendations(customer_id=customer_id, recent_items=None, top_k=3)
    # Build readable bullet lines for product suggestions
    bullets = "\n".join([f"- {x['description']} (Code: {x['stockcode']})" for x in rec["recommendations"]])
    # Construct the full body content string
    body = (
        f"Hi Customer {customer_id},\n\n"
        f"As one of our {seg['segment_name']}, we thought you would love these picks:\n"
        f"{bullets}\n\n"
        f"Deal: {camp['proposed_offer']}\n"
        f"Shop now and enjoy a tailored experience.\n\n"
        f"— Your Favorite Store"
    )
    # Return the subject and body ready to be sent by an ESP
    return {"customer_id": customer_id, "subject": subject, "body": body}

# Define the predictive analytics endpoint to estimate repeat purchase probability
@app.get("/predictive_analytics")
def predictive_analytics(customer_id: int) -> Dict[str, Any]:
    # Access the RFM row for this customer
    row = _get_customer_rfm(customer_id)
    # Build a 2D array of features for the classifier
    X = row[ARTIFACTS["rfm_features"]].values.reshape(1, -1)
    # Apply the scaler to match training space
    Xs = ARTIFACTS["rfm_scaler"].transform(X)
    # Predict probability for the positive class (repeat purchaser)
    prob = float(ARTIFACTS["repeat_clf"].predict_proba(Xs)[0, 1])
    # Define a simple policy engine to map probability to an action
    if prob >= 0.7:
        # High probability customers get loyalty upsell
        action = "Upsell premium with points bonus"
    elif prob >= 0.4:
        # Mid probability customers get modest incentive
        action = "Send 10% voucher to accelerate purchase"
    else:
        # Low probability customers get nurturing content
        action = "Retarget with educational content and social proof"
    # Return the probability and recommended action
    return {"customer_id": customer_id, "repeat_purchase_probability": round(prob, 3), "recommended_action": action}

# Define the customer journey mapping endpoint to show timeline and next best action
@app.get("/journey_mapping")
def journey_mapping(customer_id: int) -> Dict[str, Any]:
    # Retrieve precomputed journey steps for the customer
    journeys: Dict[int, List[Dict[str, Any]]] = ARTIFACTS["journeys"]
    # Validate presence of journey data
    if customer_id not in journeys:
        # Raise not found if missing
        raise HTTPException(status_code=404, detail=f"No journey found for {customer_id}")
    # Read the chronological list of events for this customer
    events = journeys[customer_id]
    # Heuristic to define a next-best-action based on last event
    last = events[-1] if events else {}
    # Determine next action based on last known touchpoint
    if last.get("event_type") == "cart_abandon":
        # Recommend win-back email with discount
        nba = "Send cart recovery email with 10% off"
    elif last.get("event_type") == "product_view":
        # Suggest push about viewed item restock or price drop
        nba = "Push notification about price drop on viewed item"
    else:
        # Fall back to general engagement nudging
        nba = "Email roundup of trending products this week"
    # Return journey events and next best action
    return {"customer_id": customer_id, "events": events, "next_best_action": nba}
