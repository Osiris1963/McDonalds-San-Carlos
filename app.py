# app.py
import streamlit as st
import pandas as pd
import time
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- Import from our re-architected modules ---
from data_processing import load_from_firestore
from forecasting import generate_customer_forecast, generate_atv_forecast

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Sales Forecaster v5.2 (Patched)",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def apply_custom_styling():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
        .main > div { background-color: #1a1a1a; }
        .block-container { padding: 2.5rem 2rem !important; }
        [data-testid="stSidebar"] { background-color: #252525; border-right: 1px solid #444; }
        .stButton > button {
            border-radius: 8px; font-weight: 600; transition: all 0.2s ease-in-out;
            border: none; padding: 10px 16px;
        }
        .stButton:has(button:contains("Generate")), .stButton:has(button:contains("Save")) > button {
            background: linear-gradient(45deg, #c8102e, #e01a37); color: #FFFFFF;
        }
        .stButton:has(button:contains("Generate")):hover > button, .stButton:has(button:contains("Save")):hover > button {
            transform: translateY(-2px); box-shadow: 0 4px 15px 0 rgba(200, 16, 46, 0.4);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px; background-color: transparent; color: #d3d3d3;
            padding: 8px 14px; font-weight: 600; font-size: 0.9rem;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #c8102e; color: #ffffff; }
        .st-expander {
            border: 1px solid #444 !important; box-shadow: none; border-radius: 10px;
            background-color: #252525; margin-bottom: 0.5rem;
        }
        .st-expander header { font-size: 0.9rem; font-weight: 600; color: #d3d3d3; }
        .stPlotlyChart { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization & Data Functions ---
@st.cache_resource
def init_firestore():
    """Initializes a connection to Firestore using Streamlit Secrets."""
    try:
        if not firebase_admin._apps:
            creds_dict = st.secrets.firebase_credentials.to_dict()
            creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: Failed to initialize. Check your Streamlit Secrets. Details: {e}")
        return None

def save_forecast_to_log(db_client, forecast_df):
    """Saves the final generated forecast to the 'forecast_log' collection."""
    if db_client is None or forecast_df.empty:
        st.warning("Database client not available or forecast is empty. Skipping log.")
        return False
    try:
        batch = db_client.batch()
        log_collection_ref = db_client.collection('forecast_log')
        generated_on_ts = pd.to_datetime('today', utc=True)

        for _, row in forecast_df.iterrows():
            doc_id = row['ds'].strftime('%Y-%m-%d')
            log_doc_ref = log_collection_ref.document(doc_id)
            
            log_data = {
                'generated_on': generated_on_ts,
                'forecast_for_date': row['ds'],
                'predicted_sales': float(row['forecast_sales']),
                'predicted_customers': int(row['forecast_customers']),
                'predicted_atv': float(row['forecast_atv'])
            }
            batch.set(log_doc_ref, log_data, merge=True)
        
        batch.commit()
        return True
    except Exception as e:
        st.error(f"Error logging forecast to database: {e}")
        return False

@st.cache_data
def get_historical_data(_db_conn):
    return load_from_firestore(_db_conn, 'historical_data')

@st.cache_data
def get_events_data(_db_conn):
    return load_from_firestore(_db_conn, 'future_activities')

def render_historical_record(row, db_client):
    """Renders an editable historical data record with data integrity checks."""
    if 'doc_id' not in row or pd.isna(row['doc_id']):
        return

    date_str = row['date'].strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    
    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}")

        day_type_options = ["Normal Day", "Not Normal Day"]
        current_day_type = row.get('day_type', day_type_options[0])

        try:
            current_index = day_type_options.index(current_day_type)
        except ValueError:
            current_index = 0
            current_day_type = day_type_options[0]
            st.warning(f"Found an invalid 'day_type' for {date_str}. Defaulting to 'Normal Day'.")

        st.write(f"**Day Type:** {current_day_type}")
        
        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            st.markdown("**Edit Record**")
            
            updated_day_type = st.selectbox(
                "Day Type", 
                day_type_options, 
                index=current_index, 
                key=f"day_type_{row['doc_id']}"
            )
            
            if st.form_submit_button("💾 Update Day Type", use_container_width=True):
                db_client.collection('historical_data').document(row['doc_id']).update({'day_type': updated_day_type})
                st.success(f"Record for {date_str} updated!")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()

# --- Main Application ---
apply_custom_styling()
db = init_firestore()

if db:
    if 'customer_forecast_df' not in st.session_state:
        st.session_state.customer_forecast_df = None
    if 'atv_forecast_df' not in st.session_state:
        st.session_state.atv_forecast_df = None
    if 'final_forecast_df' not in st.session_state:
        st.session_state.final_forecast_df = None
    if 'customer_model' not in st.session_state:
        st.session_state.customer_model = None

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v5.2")
        st.info("Decoupled Engine: LGBM + Prophet")

        if st.button("🔄 Refresh Data & Clear Cache"):
            st.cache_data.clear(); st.cache_resource.clear()
            for key in ['customer_forecast_df', 'atv_forecast_df', 'final_forecast_df', 'customer_model']:
                st.session_state[key] = None
            st.success("Caches & state cleared. Rerunning..."); time.sleep(1); st.rerun()

        st.markdown("---")
        st.subheader("Step 1: Forecast Customers")
        if st.button("📊 Forecast Customers", use_container_width=True):
            historical_df = get_historical_data(db)
            events_df = get_events_data(db)
            if len(historical_df) < 30: 
                st.error("Need at least 30 days of data.")
            else:
                with st.spinner("🧠 Training Customer Model (LGBM)..."):
                    cust_df, cust_model = generate_customer_forecast(historical_df, events_df)
                    st.session_state.customer_forecast_df = cust_df
                    st.session_state.customer_model = cust_model
                    st.session_state.final_forecast_df = None
                st.success("Customer forecast complete!")

        st.subheader("Step 2: Forecast ATV")
        if st.button("📈 Forecast ATV", use_container_width=True):
            historical_df = get_historical_data(db)
            events_df = get_events_data(db)
            if len(historical_df) < 30: 
                st.error("Need at least 30 days of data.")
            else:
                with st.spinner("⏳ Training ATV Model (Prophet)..."):
                    atv_df, _ = generate_atv_forecast(historical_df, events_df)
                    st.session_state.atv_forecast_df = atv_df
                    st.session_state.final_forecast_df = None
                st.success("ATV forecast complete!")
        
        st.markdown("---")
        st.subheader("Step 3: Generate Final Forecast")
        is_disabled = st.session_state.customer_forecast_df is None or st.session_state.atv_forecast_df is None
        if st.button("Generate Final Forecast & Save", type="primary", use_container_width=True, disabled=is_disabled):
            with st.spinner("Combining forecasts and saving..."):
                cust_df = st.session_state.customer_forecast_df
                atv_df = st.session_state.atv_forecast_df
                final_df = pd.merge(cust_df, atv_df, on='ds')
                final_df['forecast_sales'] = final_df['forecast_customers'] * final_df['forecast_atv']
                st.session_state.final_forecast_df = final_df
                
                if save_forecast_to_log(db, final_df):
                    st.success("Final Forecast Generated & Saved!")
                else:
                    st.warning("Final forecast generated but failed to save.")
        if is_disabled:
            st.caption("Complete Steps 1 & 2 to enable.")

    tab_list = ["🔮 Forecast Dashboard", "💡 Customer Model Insights", "✍️ Edit Data"]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        st.header("🔮 Forecast Dashboard")
        if st.session_state.final_forecast_df is not None:
            df = st.session_state.final_forecast_df
            st.subheader("Final Combined Sales Forecast")
            df_display = df.rename(columns={
                'ds': 'Date', 'forecast_customers': 'Predicted Customers',
                'forecast_atv': 'Predicted Avg Sale (₱)', 'forecast_sales': 'Predicted Sales (₱)'
            }).set_index('Date')
            df_display['Predicted Sales (₱)'] = df_display['Predicted Sales (₱)'].apply(lambda x: f"₱{x:,.2f}")
            df_display['Predicted Avg Sale (₱)'] = df_display['Predicted Avg Sale (₱)'].apply(lambda x: f"₱{x:,.2f}")
            st.dataframe(df_display, use_container_width=True, height=560)
        elif st.session_state.customer_forecast_df is None and st.session_state.atv_forecast_df is None:
            st.info("Begin by generating a forecast using the controls in the sidebar.")
        else:
            st.info("Final forecast not yet generated. Showing individual model outputs below.")
            c1, c2 = st.columns(2)
            if st.session_state.customer_forecast_df is not None:
                with c1:
                    st.subheader("Customer Forecast (LGBM)")
                    st.dataframe(st.session_state.customer_forecast_df, use_container_width=True)
            if st.session_state.atv_forecast_df is not None:
                with c2:
                    st.subheader("ATV Forecast (Prophet)")
                    st.dataframe(st.session_state.atv_forecast_df, use_container_width=True)

    with tabs[1]:
        st.header("💡 Key Customer Drivers (LGBM)")
        st.info("This shows the most important factors the AI model used to predict customer traffic.")
        if st.session_state.customer_model:
            model = st.session_state.customer_model
            feature_importances = pd.DataFrame({
                'feature': model.feature_name_, 'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.barplot(x='importance', y='feature', data=feature_importances, ax=ax, palette='viridis')
            ax.set_title('Top 20 Features Driving Customer Forecast', fontsize=16, color='white')
            ax.set_xlabel('Importance', fontsize=12, color='white'); ax.set_ylabel('Feature', fontsize=12, color='white')
            for spine in ax.spines.values(): spine.set_edgecolor('#555555')
            ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Generate the customer forecast to see the key drivers.")

    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        st.info("Correct the 'Day Type' for past dates to improve future forecasts.")
        historical_df_edit = get_historical_data(db)
        if not historical_df_edit.empty:
            recent_df = historical_df_edit.sort_values(by="date", ascending=False).head(30)
            for _, row in recent_df.iterrows():
                render_historical_record(row, db)
        else:
            st.info("No historical data found.")
else:
    st.error("Could not connect to Firestore. Please check your configuration and network.")
