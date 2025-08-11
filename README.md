# Customer Segmentation & Personalization (Demo)

This repository contains a minimal demo scaffold:

- FastAPI backend (backend/)
- Streamlit frontend (frontend/)
- Sample Online Retail II dataset (backend/online_retail_II.xlsx)

## Run backend

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

## Run frontend (in a separate terminal)

cd frontend
pip install streamlit requests
streamlit run dashboard.py

Default API_URL in Streamlit expects the backend at http://localhost:8000

## File structure

```
customer_segmentation_app
├── backend
│   ├── data
│   │   └── online_retail_II.xlsx
│   ├── dynamic_content.py
│   ├── email_personalization.py
│   ├── journey_mapping.py
│   ├── main.py
│   ├── marketing.py
│   ├── models
│   │   ├── personalization.py
│   │   └── __init__.py
│   ├── predictive_analytics.py
│   ├── recommendations.py
│   ├── segmentation.py
│   ├── utils.py
│   └── utils.py
├── frontend
│   ├── dashboard.py
│   └── requirements.txt
└── README.md
```
