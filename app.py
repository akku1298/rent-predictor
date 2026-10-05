import os

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="India House Rent Predictor", page_icon="🏠")
st.title("🏠 India House Rent Predictor")
st.write("Enter house details for any state in India to get estimated monthly rent.")

@st.cache_resource
def load_model():
    df = pd.read_csv("india_rent.csv")
    X = pd.get_dummies(df.drop(columns=["rent"]), drop_first=True)
    y = df["rent"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))
    return model, list(X.columns), df, mae, r2

model, feature_cols, df, mae, r2 = load_model()

st.sidebar.metric("Model MAE (Rs)", f"{int(mae)}")
st.sidebar.metric("Model R2", f"{r2:.3f}")
st.sidebar.caption(f"Trained on {len(df)} rows, {df['state'].nunique()} states/UTs")

states = sorted(df["state"].unique())
tiers = sorted(df["city_tier"].unique())
furnishings = sorted(df["furnishing"].unique())

col1, col2 = st.columns(2)
with col1:
    state = st.selectbox("State / UT", states, index=states.index("Maharashtra"))
    city_tier = st.selectbox("City type", tiers, index=tiers.index("Urban"))
    furnishing = st.selectbox("Furnishing", furnishings)
with col2:
    area = st.slider("Area (sqft)", 400, 2000, 950, step=50)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4], index=1)
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3], index=1)

if st.button("Predict Rent", type="primary"):
    inp = pd.DataFrame([{
        "state": state, "city_tier": city_tier, "area_sqft": area,
        "bedrooms": bedrooms, "bathrooms": bathrooms, "furnishing": furnishing
    }])
    X_new = pd.get_dummies(inp)
    # align with training columns
    for c in feature_cols:
        if c not in X_new.columns:
            X_new[c] = 0
    X_new = X_new[feature_cols]
    pred = model.predict(X_new)[0]
    st.session_state["pred"] = float(pred)
    st.session_state["context"] = f"{bedrooms}BHK {area}sqft in {state}, {city_tier}, {furnishing}"
    st.session_state["state_avg"] = int(df[df["state"] == state]["rent"].mean())
    st.session_state["state_name"] = state

if "pred" in st.session_state:
    st.success(f"Estimated monthly rent: Rs {int(st.session_state['pred']):,}")
    st.info(f"Average rent in {st.session_state['state_name']} (dataset): Rs {st.session_state['state_avg']:,}")

    if st.button("Explain with AI"):
        with st.spinner("Asking Gemini..."):
            try:
                try:
                    if "GEMINI_API_KEY" in st.secrets:
                        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
                except Exception:
                    pass  # no secrets file, fall back to env var
                from llm import chat
                st.write(chat(f"Explain Rs {int(st.session_state['pred'])} rent for {st.session_state['context']} in 3 bullets."))
            except Exception as e:
                st.error(f"AI error: {e}")
