from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class PredictiveAnalyticsRequest(BaseModel):
    data: List[Dict]

router = APIRouter()

@router.post("/predictive_analytics")
def predictive_analytics(request: PredictiveAnalyticsRequest):
    df = pd.DataFrame(request.data)
    df = df.dropna(subset=['Customer ID'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    customer_summary = df.groupby('Customer ID').agg({'TotalPrice': 'sum', 'Quantity': 'sum'})
    customer_summary['RepeatPurchase'] = (customer_summary['Quantity'] > 10).astype(int)

    X = customer_summary[['TotalPrice', 'Quantity']]
    y = customer_summary['RepeatPurchase']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return {
        "accuracy": accuracy,
        "feature_importance": dict(zip(X.columns, model.feature_importances_))
    }
