# app.py
import streamlit as st
import pandas as pd
import time
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import from our new, separated modules ---
from data_processing import load_from_firestore
# // SENIOR DEV NOTE //: The import itself is correct. The error was in how the function was *called*.
from forecasting import generate_forecast

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Sales Forecaster v4.2",
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

# --- Firestore Initialization ---
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
    """Saves the generated forecast to the 'forecast_log' collection in Firestore."""
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

def render_historical_record(row, db_client):
    """Renders an editable historical data record."""
    if 'doc_id' not in row or pd.isna(row['doc_id']):
        return

    date_str = row['date'].strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    
    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}")
        day_type = row.get('day_type', 'Normal Day')
        st.write(f"**Day Type:** {day_type}")
        
        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            st.markdown("**Edit Record**")
            day_type_options = ["Normal Day", "Not Normal Day"]
            current_day_type = row.get('day_type', 'Normal Day')
            current_index = day_type_options.index(current_day_type) if current_day_type in day_type_options else 0
            
            updated_day_type = st.selectbox("Day Type", day_type_options, index=current_index, key=f"day_type_{row['doc_id']}")
            
            if st.form_submit_button("💾 Update Day Type", use_container_width=True):
                update_data = {'day_type': updated_day_type}
                db_client.collection('historical_data').document(row['doc_id']).update(update_data)
                st.success(f"Record for {date_str} updated!")
                st.cache_data.clear()
                time.sleep(1); st.rerun()

# --- Main Application ---
apply_custom_styling()
db = init_firestore()

if db:
    # Initialize session state
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'customer_model' not in st.session_state:
        st.session_state.customer_model = None
    # // SENIOR DEV NOTE //: Initialize the new atv_model in the session state.
    if 'atv_model' not in st.session_state:
        st.session_state.atv_model = None


    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v4.2")
        st.info("Unified Engine - Dual Weighting")

        if st.button("🔄 Refresh Data & Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared. Rerunning...")
            time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df = load_from_firestore(db, 'historical_data')
            events_df = load_from_firestore(db, 'future_activities')

            if len(historical_df) < 30: 
                st.error("Need at least 30 days of data for a reliable forecast.")
            else:
                with st.spinner("🧠 Training Adaptive Forecasting Engine..."):
                    # // SENIOR DEV NOTE //: THIS IS THE FIX.
                    # Unpack all THREE return values from the updated function.
                    forecast_df, customer_model, atv_model = generate_forecast(historical_df, events_df, periods=15)
                    
                    # Store all results in the session state
                    st.session_state.forecast_df = forecast_df
                    st.session_state.customer_model = customer_model
                    st.session_state.atv_model = atv_model
                
                if not forecast_df.empty:
                    with st.spinner("📡 Logging forecast to database..."):
                        save_successful = save_forecast_to_log(db, forecast_df)
                    
                    if save_successful:
                        st.success("Forecast Generated & Logged!")
                    else:
                        st.warning("Forecast generated but failed to log.")
                else:
                    st.error("Forecast generation failed. Check data for errors.")

    tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "✍️ Edit Data"]
    tabs = st.tabs(tab_list)

    # --- Forecast Dashboard Tab ---
    with tabs[0]:
        st.header("🔮 Forecast Dashboard")
        if not st.session_state.forecast_df.empty:
            df_to_show = st.session_state.forecast_df.rename(columns={
                'ds': 'Date', 'forecast_customers': 'Predicted Customers',
                'forecast_atv': 'Predicted Avg Sale (₱)', 'forecast_sales': 'Predicted Sales (₱)'
            }).set_index('Date')
            
            df_to_show['Predicted Sales (₱)'] = df_to_show['Predicted Sales (₱)'].apply(lambda x: f"₱{x:,.2f}")
            df_to_show['Predicted Avg Sale (₱)'] = df_to_show['Predicted Avg Sale (₱)'].apply(lambda x: f"₱{x:,.2f}")
            
            st.dataframe(df_to_show, use_container_width=True, height=560)
        else:
            st.info("Click 'Generate Forecast' in the sidebar to begin.")

    # --- Forecast Insights Tab ---
    with tabs[1]:
        st.header("💡 Key Forecast Drivers")
        st.info("This shows the most important factors the AI used to predict outcomes.")

        # // SENIOR DEV NOTE //: Upgraded this tab to show insights for BOTH models.
        if st.session_state.customer_model and st.session_state.atv_model:
            model_choice = st.radio(
                "Select Model to Analyze:",
                ('Customer Model', 'ATV Model'),
                horizontal=True
            )

            model, title_text = (st.session_state.customer_model, 'Top 20 Features Driving Customer Forecast') if model_choice == 'Customer Model' else (st.session_state.atv_model, 'Top 20 Features Driving ATV Forecast')

            feature_importances = pd.DataFrame({
                'feature': model.feature_name_,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 10))
            
            sns.barplot(x='importance', y='feature', data=feature_importances, ax=ax, palette='viridis')
            ax.set_title(title_text, fontsize=16)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#555555')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            fig.tight_layout()
            
            st.pyplot(fig)
        else:
            st.info("Generate a forecast to see the key drivers of customer and spending behavior.")

    # --- Edit Data Tab ---
    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        st.info("Correct the 'Day Type' for past dates to improve future forecasts.")
        
        @st.cache_data
        def get_historical_data(_db_conn):
            """
            This function takes a dummy argument to associate it with the DB connection.
            Streamlit can now safely cache the returned DataFrame.
            """
            return load_from_firestore(_db_conn, 'historical_data')

        historical_df_edit = get_historical_data(db)
        
        if not historical_df_edit.empty:
            recent_df = historical_df_edit.sort_values(by="date", ascending=False).head(30)
            for _, row in recent_df.iterrows():
                render_historical_record(row, db)
        else:
            st.info("No historical data found.")
else:
    st.error("Could not connect to Firestore. Please check your configuration and network.")
