import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import aggregate_customer_summary

# Train segmentation model on import
_rfm = aggregate_customer_summary()
_features = _rfm[["Recency", "Frequency", "Monetary"]].fillna(0)
_scaler = StandardScaler()
_X = _scaler.fit_transform(_features)
_kmeans = KMeans(n_clusters=4, random_state=42).fit(_X)
_rfm["SegmentID"] = _kmeans.labels_

# Map to human-readable labels by ordering clusters by Monetary
_order = _rfm.groupby("SegmentID")["Monetary"].median().sort_values().index.tolist()
_label_map = {seg: label for seg, label in zip(_order, ["Low", "Mid", "High", "VIP"]) }
_rfm["Segment"] = _rfm["SegmentID"].map(_label_map)
_rfm_indexed = _rfm.set_index("Customer ID")


def segment_customers(data):
    """If customer_id provided, return their segment; otherwise return all customers' segments."""
    if data.customer_id:
        cid = data.customer_id
        if cid in _rfm_indexed.index:
            row = _rfm_indexed.loc[cid]
            return {
                "customer_id": int(cid),
                "recency": int(row["Recency"]),
                "frequency": int(row["Frequency"]),
                "monetary": float(row["Monetary"]),
                "segment": str(row["Segment"]),
            }
        else:
            return {"error": "Customer not found"}
    else:
        return _rfm.reset_index()[["Customer ID", "Recency", "Frequency", "Monetary", "Segment"]].to_dict(orient="records")