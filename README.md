# Customer Segmentation & Personalization

## Quickstart

1. `python -m venv .venv && source .venv/bin/activate` (or activate on Windows).
2. `pip install -r requirements.txt`
3. `python data/make_sample_data.py` # creates data/online_retail_II_sample.csv
4. `python backend/train.py` # trains and exports artifacts
5. `uvicorn backend.main:app --reload` # starts the API at http://127.0.0.1:8000
6. Create `.streamlit/secrets.toml` in `frontend/` with `API_URL="http://127.0.0.1:8000"`.
7. `streamlit run frontend/dashboard.py` # launches the UI.

## File structure

backend/
└── main.py
└── train.py
└── models/
└── (generated .pkl files after training)
data/
└── make_sample_data.py
└── online_retail_II.xlsx
frontend/
└── dashboard.py
requirements.txt
README.md
