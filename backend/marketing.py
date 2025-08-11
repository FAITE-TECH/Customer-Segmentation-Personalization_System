from utils import load_data, aggregate_customer_summary
import pandas as pd

rfm = aggregate_customer_summary()

CAMPAIGN_LIBRARY = {
    "VIP": [
        {"title": "Exclusive Preview Sale", "desc": "Invite to 24h preview + 15% off"},
        {"title": "Loyalty Gift", "desc": "Free gift for top spenders"},
    ],
    "High": [
        {"title": "Free Shipping Over $50", "desc": "Encourage more frequent purchases"},
        {"title": "Product Bundle Offer", "desc": "Bundle complementary items at discount"},
    ],
    "Mid": [
        {"title": "Welcome Back Discount", "desc": "10% off if they purchase in 14 days"},
        {"title": "Cross-sell Recommendation", "desc": "Suggested items based on past purchases"},
    ],
    "Low": [
        {"title": "Discount Coupon", "desc": "20% off to win back price-sensitive shoppers"},
    ],
}


def personalized_campaign(data):
    seg = data.customer_segment
    seg = seg if seg in CAMPAIGN_LIBRARY else "Mid"

    # Pick top 2 campaigns for segment
    campaigns = CAMPAIGN_LIBRARY.get(seg, CAMPAIGN_LIBRARY["Mid"])[:2]

    # Add a simple uplift estimate using average monetary
    avg_monetary = rfm[rfm["Customer ID"].isin(rfm["Customer ID"])].groupby("Customer ID")["Monetary"].mean().mean()

    return {
        "segment": seg,
        "campaigns": campaigns,
        "expected_lift_estimate_pct": round(5 + (0 if seg == "Low" else (5 if seg == "Mid" else (10 if seg == "High" else 15))), 2),
        "notes": "Personalize subject lines and timing using customer's local timezone when available",
    }