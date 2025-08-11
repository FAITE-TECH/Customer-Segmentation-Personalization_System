import pandas as pd
from pathlib import Path
from dateutil import parser

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "online_retail_II.xlsx"

# Simple cache
_cache = {}


def load_data():
    """Load dataset, basic cleaning and feature engineering."""
    if "df" in _cache:
        return _cache["df"].copy()

    df = pd.read_excel(DATA_PATH, engine="openpyxl")

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Drop rows missing Customer ID or Invoice
    df = df.dropna(subset=["Customer ID", "Invoice"])  # Invoice sometimes empty

    # Ensure types
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["Quantity"] = df["Quantity"].astype(int)
    df["Price"] = df["Price"].astype(float)

    # Parse dates
    try:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
    except Exception:
        df["InvoiceDate"] = df["InvoiceDate"].apply(lambda x: parser.parse(str(x)))

    # Compute total price
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    # Sort
    df = df.sort_values(["Customer ID", "InvoiceDate"]).reset_index(drop=True)

    _cache["df"] = df
    return df.copy()


def get_customer_transactions(customer_id: int):
    df = load_data()
    return df[df["Customer ID"] == customer_id].copy()


def aggregate_customer_summary():
    df = load_data()
    # Aggregate: Recency, Frequency, Monetary (RFM)
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice": "nunique",
        "TotalPrice": "sum",
    }).reset_index()
    rfm.columns = ["Customer ID", "Recency", "Frequency", "Monetary"]

    # Fill zeros
    rfm["Monetary"] = rfm["Monetary"].fillna(0)
    return rfm