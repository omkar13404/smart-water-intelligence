#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Smart Water Intelligence Platform",
    layout="wide"
)

# -----------------------------------
# LOAD MODELS
# -----------------------------------
import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

collection_model = pickle.load(
    open(
        os.path.join(
            MODEL_DIR,
            "collection_model.pkl"
        ),
        "rb"
    )
)

overflow_model = pickle.load(
    open(
        os.path.join(
            MODEL_DIR,
            "overflow_model.pkl"
        ),
        "rb"
    )
)

anomaly_model = pickle.load(
    open(
        os.path.join(
            MODEL_DIR,
            "anomaly_model.pkl"
        ),
        "rb"
    )
)


# -----------------------------------
# LOAD DATA
# -----------------------------------

DATA_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data"
)

df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "smart_water_dataset.csv"
    )
)

forecast_df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "forecast_results.csv"
    )
)
# -----------------------------------
# TITLE
# -----------------------------------

st.title(
    "Smart Sustainable Water Intelligence Platform"
)

st.markdown(
    """
AI-powered platform for:
- Water forecasting
- Overflow prediction
- Leakage detection
- Sustainability analytics
"""
)

# -----------------------------------
# SIDEBAR INPUTS
# -----------------------------------

st.sidebar.header("Input Parameters")

season = st.sidebar.selectbox(
    "Season",
    ["Monsoon", "Summer", "Winter"]
)

season_map = {
    "Monsoon": 0,
    "Summer": 1,
    "Winter": 2
}

season_encoded = season_map[season]

rainfall = st.sidebar.slider(
    "Rainfall (mm)",
    0,
    150,
    50
)

humidity = st.sidebar.slider(
    "Humidity",
    0,
    100,
    70
)

temperature = st.sidebar.slider(
    "Temperature",
    10,
    45,
    28
)

tank_capacity = st.sidebar.selectbox(
    "Tank Capacity",
    [1000, 1500, 2000]
)

water_level = st.sidebar.slider(
    "Current Water Level",
    0,
    tank_capacity,
    500
)

daily_usage = st.sidebar.slider(
    "Daily Usage",
    50,
    700,
    200
)

occupancy = st.sidebar.slider(
    "Occupancy",
    1,
    15,
    5
)

roof_area = st.sidebar.slider(
    "Roof Area",
    50,
    250,
    120
)

efficiency = st.sidebar.slider(
    "Collection Efficiency",
    0.5,
    1.0,
    0.8
)

# -----------------------------------
# CREATE INPUT ARRAY
# -----------------------------------

features = np.array([
    [
        season_encoded,
        rainfall,
        humidity,
        temperature,
        tank_capacity,
        water_level,
        daily_usage,
        occupancy,
        roof_area,
        efficiency
    ]
])

# -----------------------------------
# PREDICTIONS
# -----------------------------------

collection_prediction = (
    collection_model.predict(features)[0]
)

overflow_prediction = (
    overflow_model.predict(features)[0]
)

overflow_probability = (
    overflow_model.predict_proba(features)[0][1]
)

# -----------------------------------
# ANOMALY DETECTION INPUT
# -----------------------------------

anomaly_features = np.array([
    [
        season_encoded,
        rainfall,
        humidity,
        temperature,
        water_level,
        daily_usage,
        occupancy,
        collection_prediction
    ]
])

anomaly_prediction = (
    anomaly_model.predict(anomaly_features)[0]
)

# -----------------------------------
# KPI SECTION
# -----------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Predicted Collection",
        f"{collection_prediction:.2f} L"
    )

with col2:

    st.metric(
        "Overflow Probability",
        f"{overflow_probability * 100:.2f}%"
    )

with col3:

    sustainability_score = max(
        0,
        min(
            100,
            100 - (daily_usage / 5)
            + (collection_prediction / 50)
        )
    )

    st.metric(
        "Sustainability Score",
        f"{sustainability_score:.2f}"
    )

with col4:

    st.metric(
        "Current Tank Level",
        f"{water_level} L"
    )

# -----------------------------------
# ALERTS
# -----------------------------------

st.subheader("System Alerts")

if overflow_prediction == 1:

    st.error(
        "WARNING: High Overflow Risk!"
    )

else:

    st.success(
        "Tank Status: SAFE"
    )

if anomaly_prediction == -1:

    st.warning(
        "Possible Leakage / Abnormal Usage Detected!"
    )

else:

    st.success(
        "No Leakage Detected"
    )

# -----------------------------------
# FORECAST GRAPH
# -----------------------------------

st.subheader(
    "Future Water Collection Forecast"
)

fig = px.line(
    forecast_df,
    x="ds",
    y="yhat",
    title="Forecasted Water Collection"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------
# DATA ANALYTICS
# -----------------------------------

st.subheader(
    "Historical Analytics"
)

fig2 = px.scatter(
    df,
    x="water_level",
    y="daily_usage",
    color="overflow_risk",
    title="Water Usage vs Water Level"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# -----------------------------------
# SEASONAL ANALYSIS
# -----------------------------------

st.subheader(
    "Seasonal Water Collection"
)

fig3 = px.box(
    df,
    x="season",
    y="water_collected",
    color="season"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# -----------------------------------
# RECOMMENDATIONS
# -----------------------------------

st.subheader(
    "AI Recommendations"
)

if rainfall > 80:

    st.info(
        "High rainfall expected. "
        "Increase storage preparation."
    )

if sustainability_score < 50:

    st.warning(
        "Water usage efficiency is low."
    )

if daily_usage > 400:

    st.warning(
        "Daily usage unusually high."
    )

if overflow_probability > 0.8:

    st.error(
        "Immediate overflow mitigation recommended."
    )

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.caption(
    "Developed using Machine Learning, "
    "Forecasting, Anomaly Detection, "
    "and Sustainability Analytics"
)


# In[ ]:




