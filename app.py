# app.py
import streamlit as st
import pandas as pd
import time
from datetime import date
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objs as go

# --- Import from our new modules ---
from data_processing import load_from_firestore
from forecasting import generate_forecast

# --- Page Configuration and Styling (Keep your existing CSS) ---
st.set_page_config(page_title="Sales Forecaster", layout="wide")
st.markdown("""<style> ... your CSS here ... </style>""", unsafe_allow_html=True) # Keep your styling

# --- Firestore Initialization (Keep your existing function) ---
@st.cache_resource
def init_firestore():
    # ... your existing firestore init code ...
    try:
        if not firebase_admin._apps:
            # Your secrets logic here
            cred = credentials.Certificate(...) 
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: {e}")
        return None

# --- Main App ---
db = init_firestore()

if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()

if db:
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v2.0")
        st.info("Re-architected for precision and speed.")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df = load_from_firestore(db, 'historical_data')
            events_df = load_from_firestore(db, 'future_activities')

            if len(historical_df) < 50:
                st.error("Need at least 50 days of data for a reliable forecast.")
            else:
                with st.spinner("🧠 Running new efficient ensemble model..."):
                    forecast = generate_forecast(historical_df, events_df, periods=15)
                    st.session_state.forecast_df = forecast
                st.success("Forecast Generated!")
                # Add your forecast logging logic here if desired
    
    # --- Main content area ---
    st.header("🔮 Forecast Dashboard")
    
    if not st.session_state.forecast_df.empty:
        forecast_df = st.session_state.forecast_df
        display_df = forecast_df.rename(columns={
            'ds': 'Date',
            'forecast_customers': 'Predicted Customers',
            'forecast_atv': 'Predicted Avg Sale (₱)',
            'forecast_sales': 'Predicted Sales (₱)'
        })
        
        st.dataframe(
            display_df.set_index('Date'),
            column_config={
                "Predicted Customers": st.column_config.NumberColumn(format="%d"),
                "Predicted Avg Sale (₱)": st.column_config.NumberColumn(format="₱%.2f"),
                "Predicted Sales (₱)": st.column_config.NumberColumn(format="₱%.2f"),
            },
            use_container_width=True,
            height=560
        )
        
        # Add your plots and other tabs here, calling the cleaner data
        # Example Plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['forecast_sales'], name='Sales Forecast'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['forecast_customers'], name='Customer Forecast', yaxis='y2'))
        fig.update_layout(
            title='15-Day Sales & Customer Forecast',
            yaxis=dict(title='Predicted Sales (₱)'),
            yaxis2=dict(title='Predicted Customers', overlaying='y', side='right'),
            paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Click 'Generate Forecast' in the sidebar to begin.")

# You can add back your other tabs ('Add/Edit Data', 'Future Activities', etc.) here.
# Their logic remains largely the same, but now the forecasting part is separate and robust.
