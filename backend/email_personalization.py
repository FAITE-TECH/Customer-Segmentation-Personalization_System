from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from fastapi import APIRouter

router = APIRouter()

class EmailRequest(BaseModel):
    customer_id: int
    data: List[Dict]  # list of rows in JSON

@router.post("/email-personalization")
def personalize_email(request: EmailRequest):
    df = pd.DataFrame(request.data)
    customer_data = df[df['Customer ID'] == request.customer_id]
    if customer_data.empty:
        return {"error": "Customer not found"}
    # favorite_item = customer_data['Description'].mode()[0]
    subject = f"Hey {request.customer_id}, Your Favorite Item is Back in Stock!"
    body = f"We noticed you loved Popular Item A. It's back in stock — grab it before it’s gone!"
    return {"subject": subject, "body": body}
