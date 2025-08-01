import streamlit as st
import pandas as pd
import time
from datetime import date, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import mean_absolute_error

# --- Import from our new, separated modules ---
# Ensure data_processing.py and forecasting.py are in the same directory
from data_processing import load_from_firestore
from forecasting import generate_forecast

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Sales Forecaster v2.0",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (from your original file) ---
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
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization (CORRECTED) ---
@st.cache_resource
def init_firestore():
    """Initializes a connection to Firestore using Streamlit Secrets."""
    try:
        if not firebase_admin._apps:
            # This is your original, correct logic for building the credentials dictionary
            creds_dict = {
              "type": st.secrets.firebase_credentials.type,
              "project_id": st.secrets.firebase_credentials.project_id,
              "private_key_id": st.secrets.firebase_credentials.private_key_id,
              "private_key": st.secrets.firebase_credentials.private_key.replace('\\n', '\n'),
              "client_email": st.secrets.firebase_credentials.client_email,
              "client_id": st.secrets.firebase_credentials.client_id,
              "auth_uri": st.secrets.firebase_credentials.auth_uri,
              "token_uri": st.secrets.firebase_credentials.token_uri,
              "auth_provider_x509_cert_url": st.secrets.firebase_credentials.auth_provider_x509_cert_url,
              "client_x509_cert_url": st.secrets.firebase_credentials.client_x509_cert_url
            }
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: Failed to initialize. Check your Streamlit Secrets. Details: {e}")
        return None

# --- Firestore Data Management Functions (from your original file) ---
def add_to_firestore(db_client, collection_name, record, existing_df):
    """Adds a new record to Firestore, checking for duplicates by date."""
    record_date = pd.to_datetime(record['date']).normalize()
    if not existing_df.empty and record_date in pd.to_datetime(existing_df['date']).dt.normalize().values:
        st.error(f"A record for {record_date.strftime('%Y-%m-%d')} already exists.")
        return
    db_client.collection(collection_name).add(record)

def delete_from_firestore(db_client, collection_name, doc_id):
    """Deletes a document from a Firestore collection."""
    db_client.collection(collection_name).document(doc_id).delete()

def update_firestore_record(db_client, collection_name, doc_id, update_data):
    """Updates a document in a Firestore collection."""
    db_client.collection(collection_name).document(doc_id).update(update_data)

# --- UI Component Rendering Functions (from your original file) ---
def render_historical_record(row, db_client):
    """Renders an editable historical data record."""
    # Defensive check for doc_id
    if 'doc_id' not in row or pd.isna(row['doc_id']):
        # st.warning(f"Skipping record for date {row['date'].strftime('%Y-%m-%d')} due to missing document ID.")
        return

    date_str = row['date'].strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    
    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}")
        st.write(f"**Day Type:** {row.get('day_type', 'Normal Day')}")
        if row.get('day_type') == 'Not Normal Day':
            st.write(f"**Notes:** {row.get('day_type_notes', 'N/A')}")

        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            st.markdown("**Edit Record**")
            edit_cols = st.columns(2)
            updated_sales = edit_cols[0].number_input("Sales (₱)", value=float(row.get('sales', 0)), key=f"sales_{row['doc_id']}")
            updated_customers = edit_cols[1].number_input("Customers", value=int(row.get('customers', 0)), key=f"cust_{row['doc_id']}")
            
            # --- ROBUST FIX FOR ValueError ---
            day_type_options = ["Normal Day", "Not Normal Day"]
            current_day_type = row.get('day_type', 'Normal Day')
            
            # If the value from the database isn't valid, default to the first option
            if current_day_type not in day_type_options:
                current_day_type = day_type_options[0] 
            
            current_index = day_type_options.index(current_day_type)
            
            updated_day_type = st.selectbox("Day Type", day_type_options, index=current_index, key=f"day_type_{row['doc_id']}")
            # --- END OF FIX ---

            updated_day_type_notes = st.text_input("Notes", value=row.get('day_type_notes', ''), key=f"notes_{row['doc_id']}")

            btn_cols = st.columns(2)
            if btn_cols[0].form_submit_button("💾 Update Record", use_container_width=True):
                update_data = {
                    'sales': updated_sales, 'customers': updated_customers,
                    'day_type': updated_day_type,
                    'day_type_notes': updated_day_type_notes if updated_day_type == 'Not Normal Day' else ''
                }
                update_firestore_record(db_client, 'historical_data', row['doc_id'], update_data)
                st.success(f"Record for {date_str} updated!")
                st.cache_data.clear()
                time.sleep(1); st.rerun()

            if btn_cols[1].form_submit_button("🗑️ Delete Record", use_container_width=True):
                delete_from_firestore(db_client, 'historical_data', row['doc_id'])
                st.warning(f"Record for {date_str} deleted.")
                st.cache_data.clear()
                time.sleep(1); st.rerun()

# --- Main Application ---
apply_custom_styling()
db = init_firestore()

if db:
    # Initialize session state
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'historical_df' not in st.session_state:
        st.session_state.historical_df = load_from_firestore(db, 'historical_data')
    if 'events_df' not in st.session_state:
        st.session_state.events_df = load_from_firestore(db, 'future_activities')

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v2.0")
        st.info("Re-architected for precision and speed.")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear()
            st.session_state.historical_df = load_from_firestore(db, 'historical_data')
            st.session_state.events_df = load_from_firestore(db, 'future_activities')
            st.success("Data refreshed.")
            time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            if len(st.session_state.historical_df) < 50:
                st.error("Need at least 50 days of data for a reliable forecast.")
            else:
                with st.spinner("🧠 Running new efficient ensemble model... (Prophet + LightGBM)"):
                    forecast = generate_forecast(
                        st.session_state.historical_df, 
                        st.session_state.events_df, 
                        periods=15
                    )
                    st.session_state.forecast_df = forecast
                st.success("Forecast Generated!")

    # --- Main Content Area with Tabs ---
    tab_list = ["🔮 Forecast Dashboard", "✍️ Add/Edit Data"]
    tabs = st.tabs(tab_list)

    # --- Forecast Dashboard Tab ---
    with tabs[0]:
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
                use_container_width=True, height=560
            )
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['forecast_sales'], name='Sales Forecast', line=dict(color='#ffc72c')))
            fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['forecast_customers'], name='Customer Forecast', yaxis='y2', line=dict(color='#c8102e')))
            fig.update_layout(
                title='15-Day Sales & Customer Forecast',
                yaxis=dict(title='Predicted Sales (₱)', color='#ffc72c'),
                yaxis2=dict(title='Predicted Customers', overlaying='y', side='right', color='#c8102e'),
                paper_bgcolor='#2a2a2a', plot_bgcolor='#2a2a2a', font_color='white',
                legend=dict(x=0.01, y=0.99, orientation='h')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click 'Generate Forecast' in the sidebar to begin.")

    # --- Add/Edit Data Tab ---
    with tabs[1]:
        st.header("✍️ Add or Edit Historical Data")
        form_col, display_col = st.columns([2, 3], gap="large")

        with form_col:
            st.subheader("Add New Daily Record")
            with st.form("new_record_form", clear_on_submit=True, border=True):
                new_date = st.date_input("Date", date.today())
                new_sales = st.number_input("Total Sales (₱)", min_value=0.0, format="%.2f")
                new_customers = st.number_input("Customer Count", min_value=0)
                new_addons = st.number_input("Add-on Sales (₱)", min_value=0.0, format="%.2f")
                new_day_type = st.selectbox("Day Type", ["Normal Day", "Not Normal Day"])
                new_day_type_notes = st.text_area("Notes (if Not Normal Day)")

                if st.form_submit_button("✅ Save Record", use_container_width=True):
                    new_rec = {
                        "date": pd.to_datetime(new_date),
                        "sales": new_sales, "customers": new_customers,
                        "add_on_sales": new_addons, "day_type": new_day_type,
                        "day_type_notes": new_day_type_notes if new_day_type == "Not Normal Day" else ""
                    }
                    add_to_firestore(db, 'historical_data', new_rec, st.session_state.historical_df)
                    st.success("Record added!")
                    st.cache_data.clear()
                    time.sleep(1); st.rerun()
        
        with display_col:
            st.subheader("Edit Recent Entries")
            df = st.session_state.historical_df.copy()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                
                # More robustly get doc_ids to prevent errors
                docs_with_ids = {doc.id: doc.to_dict() for doc in db.collection('historical_data').stream()}
                id_map = {}
                for doc_id, doc_data in docs_with_ids.items():
                    if 'date' in doc_data:
                        # Ensure timezone consistency before formatting
                        doc_date = pd.to_datetime(doc_data['date']).tz_localize(None).strftime('%Y-%m-%d')
                        id_map[doc_date] = doc_id
                
                df['doc_id'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').map(id_map)
                
                recent_df = df.sort_values(by="date", ascending=False).head(15)
                for _, row in recent_df.iterrows():
                    render_historical_record(row, db)
            else:
                st.info("No historical data to display.")
else:
    st.error("Could not connect to Firestore. Please check your configuration and network.")
