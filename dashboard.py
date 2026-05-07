import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os

# Page Config
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide", page_icon="📈")

st.title("📊 Beverage Sales Forecasting Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Controls")
state = st.sidebar.selectbox("Select US State", [
    "Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Florida", 
    "Georgia", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", 
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", 
    "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", 
    "North Carolina", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", 
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", 
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
])

# API Call
@st.cache_data(ttl=300)
def get_forecast(state_name):
    try:
        response = requests.get(f"http://127.0.0.1:8000/api/v1/forecast/{state_name}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"8-Week Forecast for {state}")
    data = get_forecast(state)
    
    if data:
        forecast_values = data['forecast']
        weeks = [f"Week {i+1}" for i in range(8)]
        
        df_plot = pd.DataFrame({
            "Timeline": weeks,
            "Predicted Sales": forecast_values
        })
        
        fig = px.line(df_plot, x="Timeline", y="Predicted Sales", 
                     title=f"Predicted Sales Trend ({data['model_used'].upper()})",
                     markers=True, template="plotly_dark")
        fig.update_traces(line_color='#00d1b2')
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"Model Used: {data['model_used'].upper()}")
        st.info(f"Request ID: {data['request_id']}")
    else:
        st.error("Could not connect to API. Please ensure FastAPI is running (uvicorn api.app:app).")

with col2:
    st.subheader("Model Performance (Latest Evaluation)")
    # Extracted from latest evaluation results
    results = {
        "Model": ["XGBoost", "Prophet", "SARIMA", "LSTM"],
        "Accuracy (%)": [97.11, 95.84, 96.60, 93.86],
        "RMSE": [12336739, 12630328, 15354727, 15806730]
    }
    df_results = pd.DataFrame(results)
    
    fig_acc = px.bar(df_results, x="Model", y="Accuracy (%)", 
                     color="Model", title="Accuracy Comparison",
                     template="plotly_dark")
    st.plotly_chart(fig_acc, use_container_width=True)
    
    st.dataframe(df_results, hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("Developed by Nancy Singh | Powered by FastAPI & Streamlit")
