# app.py
import streamlit as st
import pandas as pd
import time
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objects as go

from data_processing import load_from_firestore
from forecasting import generate_forecast

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Sales Forecaster v5.3 Final", layout="wide")

@st.cache_resource
def init_firestore():
    """
    Initializes a connection to Firestore with the final, definitive fix.
    """
    try:
        if "firebase_credentials" not in st.secrets:
            st.error("`firebase_credentials` not found in Streamlit Secrets. Please check your secrets.toml file.")
            return None

        creds_dict = st.secrets["firebase_credentials"]
        
        if not firebase_admin._apps:
            # --- THIS IS THE FIX ---
            # We must convert the Streamlit Secrets object to a real dictionary.
            cred = credentials.Certificate(dict(creds_dict))
            # --------------------
            
            firebase_admin.initialize_app(cred)
            
        return firestore.client()

    except Exception as e:
        st.error(f"""
        **Firestore Connection Error:** Failed to initialize Firebase.
        **Error Type:** `{type(e).__name__}`
        **Error Details:** {e}
        Please double-check your secrets.toml file and GCP IAM permissions.
        """)
        return None

# --- Main Application Logic ---

db = init_firestore()

if db:
    # --- The rest of your application code is unchanged and correct. ---
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'model' not in st.session_state:
        st.session_state.model = None

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v5.3")
        st.info("Robust Engine - Production Build")

        if st.button("🔄 Refresh Data & Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared. Rerunning...")
            time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df = load_from_firestore(db, 'historical_data')
            events_df = load_from_firestore(db, 'future_activities')

            if len(historical_df) < 90:
                st.error("Need at least 90 days of data for a reliable TFT forecast.")
            else:
                with st.spinner("🧠 Training Robust Forecasting Engine... (This may take a few minutes)"):
                    forecast_df, model = generate_forecast(historical_df, events_df, periods=15)
                    st.session_state.forecast_df = forecast_df
                    st.session_state.model = model
                
                if not forecast_df.empty:
                    st.success("Forecast Generated Successfully!")
                else:
                    st.error("Forecast generation failed.")

    tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "✍️ Edit Data"]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        st.header("🔮 Forecast Dashboard")
        if not st.session_state.forecast_df.empty:
            df = st.session_state.forecast_df
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['ds'], y=df['predicted_customers'], mode='lines+markers',
                line=dict(color='#c8102e', width=3), name='Predicted Customers'
            ))
            
            fig.update_layout(
                title="Customer Forecast", yaxis_title="Predicted Customers"
            )
            st.plotly_chart(fig, use_container_width=True)

            display_df = df[['ds', 'predicted_customers', 'predicted_atv', 'predicted_sales']].rename(columns={
                'ds': 'Date', 'predicted_customers': 'Predicted Customers',
                'predicted_atv': 'Predicted Avg Sale (₱)', 'predicted_sales': 'Predicted Sales (₱)'
            }).set_index('Date')
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Click 'Generate Forecast' in the sidebar to begin.")

    with tabs[1]:
        st.header("💡 Key Forecast Drivers")
        st.info("This shows the general importance of different features to the model's predictions.")
        
        if st.session_state.model:
            model = st.session_state.model
            try:
                val_dataloader = model.val_dataloader()
                importance = model.evaluate(val_dataloader, verbose=False)
                fig = model.plot_variable_importances(importance)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate feature importance plot. Details: {e}")
        else:
            st.info("Generate a forecast to see the key drivers.")
    
    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        st.info("Here you can correct the 'Day Type' for past dates. This improves future forecasts.")
        
        @st.cache_data
        def get_historical_data(_db_client):
            return load_from_firestore(_db_client, 'historical_data')

        historical_df_edit = get_historical_data(db)
        # Add logic here to display and edit historical data
        
else:
    st.warning("Could not connect to Firestore. Please check the detailed error message above and review your configuration.")
