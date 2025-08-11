import pandas as pd
from sklearn.neighbors import NearestNeighbors
from utils import load_data

_df = load_data()

# Build customer x product matrix (binary purchase)
_pivot = _df.pivot_table(index="Customer ID", columns="StockCode", values="Quantity", aggfunc="sum", fill_value=0)
_binary = (_pivot > 0).astype(int)

# Fit NearestNeighbors on item vectors (transpose to get items x customers)
_item_matrix = _binary.T  # rows: StockCode, cols: Customer ID
_model = NearestNeighbors(n_neighbors=10, metric="cosine")
_model.fit(_item_matrix.values)
_stock_codes = _item_matrix.index.tolist()
_stock_index = {code: i for i, code in enumerate(_stock_codes)}


def recommend_products(data, top_n=5):
    cid = data.customer_id
    if cid not in _binary.index:
        return {"error": "Customer not found or no purchases"}

    # Items customer bought
    bought = _binary.loc[cid]
    bought_items = bought[bought > 0].index.tolist()

    if not bought_items:
        # Cold-start: recommend top-selling products
        top = _df.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False).head(top_n)
        recs = [{"StockCode": int(idx), "score": float(val)} for idx, val in top.items()]
        return {"customer_id": cid, "recommendations": recs}

    # For each bought item find similar items
    distances = {}
    for item in bought_items:
        if item not in _stock_index:
            continue
        idx = _stock_index[item]
        _, neigh_idxs = _model.kneighbors([_item_matrix.values[idx]], n_neighbors=6)
        for nidx in neigh_idxs[0]:
            code = _stock_codes[nidx]
            if code in bought_items:
                continue
            distances[code] = distances.get(code, 0) + 1

    # Rank by co-occurrence count
    ranked = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recs = []
    for code, score in ranked:
        desc = _df[_df["StockCode"] == code]["Description"].iloc[0]
        recs.append({"StockCode": int(code), "Description": desc, "score": float(score)})

    return {"customer_id": cid, "recommendations": recs}