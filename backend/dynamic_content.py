from utils import get_customer_transactions
from segmentation import _rfm_indexed
import pandas as pd


def dynamic_content_personalization(data):
    cid = data.customer_id
    tx = get_customer_transactions
    return {
        "banner": f"Special offers curated for you, Customer {cid}!",
        "recommendations": tx(cid)["Description"].value_counts().head(3).index.tolist()
    }