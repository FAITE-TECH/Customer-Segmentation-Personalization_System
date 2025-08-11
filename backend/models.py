from pydantic import BaseModel
from typing import Optional

class SegmentationRequest(BaseModel):
    customer_id: Optional[int] = None

class MarketingRequest(BaseModel):
    customer_segment: str

class RecommendationRequest(BaseModel):
    customer_id: int

class DynamicContentRequest(BaseModel):
    customer_id: int

class EmailRequest(BaseModel):
    customer_id: int
    product_name: str

class PredictiveRequest(BaseModel):
    customer_id: int

class JourneyRequest(BaseModel):
    customer_id: int