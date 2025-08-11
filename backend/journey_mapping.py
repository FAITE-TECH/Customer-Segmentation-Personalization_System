from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd

router = APIRouter()

# Define Pydantic model for the expected input data (list of records)
class TransactionRecord(BaseModel):
    Customer_ID: int
    Invoice: str
    InvoiceDate: str
    Description: str
    Quantity: int
    Price: float

class JourneyRequest(BaseModel):
    customer_id: int
    transactions: List[TransactionRecord]

@router.post("/journey-mapping", response_model=Dict[str, Any])
async def journey_mapping(request: JourneyRequest):
    # Convert list of TransactionRecord to DataFrame
    df = pd.DataFrame([r.dict() for r in request.transactions])

    # Filter data by customer_id (note the underscore in Customer_ID)
    customer_data = df[df['Customer_ID'] == request.customer_id].sort_values(by='InvoiceDate')
    if customer_data.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    # Prepare journey data as list of dicts
    journey = customer_data[['Invoice', 'InvoiceDate', 'Description', 'Quantity', 'Price']].to_dict(orient='records')
    
    return {"customer_id": request.customer_id, "journey": journey}
