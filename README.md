# 🏠 India House Rent Predictor

Live demo: https://rent-predictor-n8mmxzb7yzwjgf7zlcxuvj.streamlit.app/

Predict average monthly house rent anywhere in India. Enter state, city type, area, BHK, furnishing and get an instant estimate.

## Features
- Covers 36 states / UTs across India
- Inputs: state, city tier (Metro / Urban / Semi-Urban / Rural), area sqft, bedrooms, bathrooms, furnishing
- Output: predicted monthly rent + state average for context
- Model: RandomForestRegressor (scikit-learn), MAE and R2 shown in sidebar

## Dataset
`india_rent.csv` — 540 rows, synthetic demo data with realistic state-wise rates:
`state,city_tier,area_sqft,bedrooms,bathrooms,furnishing,rent`

For production, replace with real listings from 99acres / MagicBricks / NoBroker.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501

## Files
- `app.py` — Streamlit app + model training
- `india_rent.csv` — training data
- `requirements.txt` — streamlit, pandas, scikit-learn
