from fastapi import FastAPI
from models import (
    SegmentationRequest,
    MarketingRequest,
    RecommendationRequest,
    DynamicContentRequest,
    EmailRequest,
    PredictiveRequest,
    JourneyRequest,
)
from segmentation import segment_customers
from marketing import personalized_campaign
from recommendations import recommend_products
from dynamic_content import dynamic_content_personalization
from email_personalization import personalize_email
from predictive_analytics import predictive_analytics
from journey_mapping import journey_mapping

app = FastAPI(title="Customer Segmentation & Personalization API")


@app.post("/segment_customers")
def segment(data: SegmentationRequest):
    return segment_customers(data)


@app.post("/personalized_marketing")
def marketing(data: MarketingRequest):
    return personalized_campaign(data)


@app.post("/product_recommendations")
def recommendations(data: RecommendationRequest):
    return recommend_products(data)


@app.post("/dynamic_content")
def dynamic_content(data: DynamicContentRequest):
    return dynamic_content_personalization(data)


@app.post("/email_personalization")
def email(data: EmailRequest):
    return personalize_email(data)


@app.post("/predictive_analytics")
def predictive(data: PredictiveRequest):
    return predictive_analytics(data)


@app.post("/journey_mapping")
def journey(data: JourneyRequest):
    return journey_mapping(data)