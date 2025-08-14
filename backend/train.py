# Import standard libraries for filesystem and typing
from pathlib import Path
from typing import Dict, List, Any, Tuple
# Import pandas and numpy for data manipulation
import pandas as pd
import numpy as np
# Import scikit-learn tools for modeling and preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
# Import joblib to persist artifacts for the API to load
import joblib

# Define paths for data input and model artifact output
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# Define the expected dataset path for Online Retail II workbook
DATA_PATH = DATA_DIR / "online_retail_II.xlsx"
# Define where to save model artifacts for the backend to consume
MODEL_DIR = Path(__file__).parent / "models"

# Ensure the output model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Define a function to read the dataset with fallback to sample data
def load_dataset() -> pd.DataFrame:
    # If the xlsx exists, read it using pandas openpyxl engine
    if DATA_PATH.exists():
        # Read the Excel file into a dataframe
        df = pd.read_excel(DATA_PATH, engine="openpyxl")
    else:
        # If missing, try to load the generated CSV fallback
        csv_path = DATA_DIR / "online_retail_II_sample.csv"
        # Read CSV if available, otherwise raise an informative error
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH} or {csv_path}. Run data/make_sample_data.py")
        # Read the CSV fallback file
        df = pd.read_csv(csv_path, parse_dates=["InvoiceDate"])
    # Return the loaded dataframe
    return df

# Define a function to clean and normalize the dataset fields
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns to consistent snake_case if needed
    df = df.rename(columns={"Customer ID": "CustomerID", "StockCode": "StockCode"})
    # Drop rows without customer identifiers as they cannot be modeled
    df = df.dropna(subset=["CustomerID"])
    # Convert identifiers to integer for index stability
    df["CustomerID"] = df["CustomerID"].astype(int)
    # Ensure InvoiceDate is a proper datetime for recency calculations
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    # Compute line total to support monetary value
    df["LineTotal"] = df["Quantity"] * df["Price"]
    # Remove negative quantities which are returns for simplicity
    df = df[df["Quantity"] > 0]
    # Return the cleaned dataframe
    return df

# Define a function to compute RFM (Recency, Frequency, Monetary) features per customer
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    # Determine the snapshot date as one day after the last transaction
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    # Group by customer and compute R, F, and M components
    recency = (snapshot - df.groupby("CustomerID")["InvoiceDate"].max()).dt.days
    frequency = df.groupby("CustomerID")["Invoice"].nunique()
    monetary = df.groupby("CustomerID")["LineTotal"].sum()
    # Combine components into a single dataframe
    rfm = pd.DataFrame({"Recency": recency, "Frequency": frequency, "Monetary": monetary})
    # Fill missing values with zeros just in case
    rfm = rfm.fillna(0)
    # Return the RFM table
    return rfm

# Define a function to create a binary label for repeat purchase modeling
def build_repeat_label(df: pd.DataFrame) -> pd.Series:
    # Count distinct invoices per customer
    inv_counts = df.groupby("CustomerID")["Invoice"].nunique()
    # Label as 1 if customer has more than one invoice (repeat buyer), else 0
    label = (inv_counts > 1).astype(int)
    # Return the label aligned by customer index
    return label

# Define a function to build a simple co-occurrence mapping for items
def build_cooccurrence(df: pd.DataFrame, top_k: int = 10) -> Dict[str, List[str]]:
    # Group by invoice to get list of items per basket
    basket_items = df.groupby("Invoice")["StockCode"].apply(list)
    # Initialize a dictionary to accumulate co-occurrence counts
    counts: Dict[Tuple[str, str], int] = {}
    # Iterate each basket to count pair co-appearances
    for items in basket_items:
        # De-duplicate items within a basket to reduce bias
        unique = list(dict.fromkeys(items))
        # For each ordered pair, increment the co-occurrence counter
        for i in range(len(unique)):
            for j in range(len(unique)):
                if i == j:
                    continue
                key = (unique[i], unique[j])
                counts[key] = counts.get(key, 0) + 1
    # Convert counts into a top-k neighbor list per item
    neighbors: Dict[str, List[str]] = {}
    for (a, b), c in counts.items():
        neighbors.setdefault(a, []).append((b, c))
    # Keep only the most frequent co-occurring items for each anchor
    result: Dict[str, List[str]] = {}
    for a, lst in neighbors.items():
        ranked = sorted(lst, key=lambda x: x[1], reverse=True)
        result[a] = [b for b, _ in ranked[:top_k]]
    # Return the mapping from item to recommended next items
    return result

# Define a function to compute product popularity per cluster for segment-aware ranking
def segment_product_popularity(df: pd.DataFrame, customer_clusters: pd.Series) -> Dict[int, List[str]]:
    # Merge cluster labels back into transaction rows
    merged = df.merge(customer_clusters.rename("Cluster"), left_on="CustomerID", right_index=True, how="inner")
    # Initialize dictionary for rankings
    ranking: Dict[int, List[str]] = {}
    # Loop over each cluster label
    for cl in sorted(merged["Cluster"].unique()):
        # Subset rows for the current cluster
        sub = merged[merged["Cluster"] == cl]
        # Count item frequency within this cluster
        top = sub["StockCode"].value_counts().index.tolist()
        # Assign ranked list of products for this cluster
        ranking[int(cl)] = top
    # Return the segment-to-products mapping
    return ranking

# Define a function to assemble per-customer journey events for the journey endpoint
def build_journeys(df: pd.DataFrame) -> Dict[int, List[Dict[str, Any]]]:
    # Sort data chronologically for deterministic timelines
    df_sorted = df.sort_values("InvoiceDate")
    # Aggregate events by customer
    journeys: Dict[int, List[Dict[str, Any]]] = {}
    # Iterate each row to build simple event logs
    for _, row in df_sorted.iterrows():
        # Initialize list for customer if missing
        journeys.setdefault(int(row["CustomerID"]), [])
        # Append a purchase event with basic details
        journeys[int(row["CustomerID"])].append(
            {
                "timestamp": row["InvoiceDate"].isoformat(),
                "event_type": "purchase",
                "invoice": str(row["Invoice"]),
                "stockcode": str(row["StockCode"]),
                "description": str(row.get("Description", "")),
                "amount": float(row["LineTotal"]),
            }
        )
    # Add a synthetic cart abandon example for demonstration (first customer only)
    if journeys:
        # Select the first key deterministically
        first_id = sorted(journeys.keys())[0]
        # Append a fabricated cart abandon event after last purchase
        journeys[first_id].append(
            {"timestamp": df_sorted["InvoiceDate"].max().isoformat(), "event_type": "cart_abandon", "invoice": "N/A"}
        )
    # Return the dictionary of journeys
    return journeys

# Define the main training routine to prepare all artifacts at once
def main() -> None:
    # Load the dataset from disk
    raw = load_dataset()
    # Clean the dataset fields and values
    df = clean_dataset(raw)
    # Build an RFM table indexed by customer identifier
    rfm = build_rfm(df)
    # Create a binary label for repeat purchase modeling
    y = build_repeat_label(df).reindex(rfm.index).fillna(0).astype(int)
    # Define the feature names to be used in modeling
    features = ["Recency", "Frequency", "Monetary"]
    # Initialize a standard scaler to normalize magnitudes
    scaler = StandardScaler()
    # Fit the scaler on the RFM features
    Xs = scaler.fit_transform(rfm[features].values)
    # Initialize a KMeans model with a small, interpretable number of clusters
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    # Fit the KMeans model on scaled RFM features
    kmeans.fit(Xs)
    # Assign cluster labels to each customer
    clusters = pd.Series(kmeans.labels_, index=rfm.index, name="Cluster")
    # Initialize a simple logistic regression classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    # Fit the classifier to predict repeat purchase from scaled RFM features
    clf.fit(Xs, y.values)
    # Build co-occurrence mapping for recommendation seeds
    cooc = build_cooccurrence(df)
    # Build popularity ranking per segment for dynamic and segment-aware recs
    seg_rank = segment_product_popularity(df, clusters)
    # Build a small product metadata dictionary for human-readable labels
    product_meta = (
        df.groupby("StockCode")["Description"]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else f"Item {s.name}")
        .to_dict()
    )
    # Construct friendly names for segments for better UX
    seg_names = {
        0: "High-Value Loyalists",
        1: "Price-Sensitive Savers",
        2: "Seasonal Buyers",
        3: "New or Lapsed",
    }
    # Save the scaler to disk for inference reuse
    joblib.dump(scaler, MODEL_DIR / "rfm_scaler.pkl")
    # Save the KMeans model to disk
    joblib.dump(kmeans, MODEL_DIR / "kmeans_segmentation.pkl")
    # Save the logistic regression classifier to disk
    joblib.dump(clf, MODEL_DIR / "repeat_purchase_clf.pkl")
    # Save the list of feature names to disk
    joblib.dump(features, MODEL_DIR / "rfm_features.pkl")
    # Save the RFM table for quick lookup in the API
    joblib.dump(rfm, MODEL_DIR / "rfm_table.pkl")
    # Save the segment-level product ranking for recs and dynamic content
    joblib.dump(seg_rank, MODEL_DIR / "segment_product_rank.pkl")
    # Save the co-occurrence mapping for item-to-item recommendations
    joblib.dump(cooc, MODEL_DIR / "cooc_recommend.pkl")
    # Save the product metadata for UI labeling
    joblib.dump(product_meta, MODEL_DIR / "product_meta.pkl")
    # Save segment label mapping for consistent naming
    joblib.dump(seg_names, MODEL_DIR / "segment_labels.pkl")
    # Build and save customer journeys for the journey mapping endpoint
    joblib.dump(build_journeys(df), MODEL_DIR / "journeys.pkl")

# Invoke the main function when running as a script
if __name__ == "__main__":
    # Execute the end-to-end training and artifact export
    main()
