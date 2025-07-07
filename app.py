import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import time
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import json
import io

# Model Imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# --- Page Configuration ---
st.set_page_config(
    page_title="McDonald's AI Forecaster",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def apply_custom_styling():
    st.markdown("""<style>
        html, body, [class*="st-"] { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        .main > div { background-color: #1e1e1e; }
        .st-emotion-cache-16txtl3 { background-color: #2a2a2a; border-right: 1px solid #444; }
        .st-emotion-cache-10trblm { color: #ffffff; }
        .stButton > button {
            border: 2px solid #c8102e; border-radius: 20px; color: #ffffff;
            background-color: #c8102e; transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover { border-color: #ffc72c; background-color: #a80d26; color: #ffc72c; }
        .stButton > button:active { border-color: #ffc72c !important; background-color: #ffc72c !important; color: #1e1e1e !important; }
        .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 8px; color: #d3d3d3; }
        .stTabs [data-baseweb="tab"]:hover { background-color: #3a3a3a; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #c8102e; color: white; font-weight: bold; }
    </style>""", unsafe_allow_html=True)

# --- Firestore Initialization ---
@st.cache_resource
def init_firestore():
    try:
        if not firebase_admin._apps:
            creds_dict = dict(st.secrets.firebase_credentials)
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: Failed to initialize. Check secrets. Error: {e}")
        return None

# --- App State Management ---
def initialize_state(db_client):
    if 'db_client' not in st.session_state:
        st.session_state.db_client = db_client
    if 'historical_df' not in st.session_state:
        st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    
    defaults = {
        'forecast_df': pd.DataFrame(), 
        'metrics': {}, 
        'name': "Store 688",
        'add_on_sales': 0.0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Firestore & Data Operations ---
def load_from_firestore(db_client, collection_name):
    if db_client is None: return pd.DataFrame()
    docs = db_client.collection(collection_name).stream()
    records = [doc.to_dict() | {'id': doc.id} for doc in docs]
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
        df.dropna(subset=['date'], inplace=True)
    return df.sort_values(by='date', ascending=True).reset_index(drop=True)

def add_to_firestore(db_client, collection_name, data):
    if db_client is None: return
    data['date'] = pd.to_datetime(data['date']).to_pydatetime()
    db_client.collection(collection_name).add(data)

def update_in_firestore(db_client, collection_name, doc_id, data):
    if db_client is None: return
    update_dict = {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}
    db_client.collection(collection_name).document(doc_id).set(update_dict, merge=True)

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Business Logic & Data Processing ---
def calculate_atv(df):
    sales = pd.to_numeric(df['sales'], errors='coerce').fillna(0)
    customers = pd.to_numeric(df['customers'], errors='coerce').fillna(0)
    df['atv'] = np.divide(sales, customers, out=np.zeros_like(sales, dtype=float), where=(customers!=0))
    return df

def calculate_dynamic_cap(df, target_col):
    recent_data = df.tail(90)
    if recent_data.empty or len(recent_data) < 10:
        return df[target_col].max() * 1.2 if not df.empty else 1000
    high_performance_level = recent_data[target_col].quantile(0.95)
    dynamic_cap = high_performance_level * 1.15
    return dynamic_cap if pd.notna(dynamic_cap) and dynamic_cap > 0 else df[target_col].max() * 1.2

def create_features(df):
    df = df.copy()
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    return df

# --- Main Forecasting Function ---
@st.cache_data
def train_and_forecast_component(historical_df, periods, target_col, model_choice, floor=0):
    df_train = historical_df.copy()
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')
    df_train.dropna(subset=['date', target_col], inplace=True)
    df_train[target_col] = pd.to_numeric(df_train[target_col], errors='coerce').ffill().bfill()

    if len(df_train) < 15: return pd.DataFrame(), {}

    cap = calculate_dynamic_cap(df_train, target_col)
    df_train['cap'] = cap
    df_train['floor'] = floor
    df_prophet = df_train.rename(columns={'date':'ds', target_col:'y'})

    prophet_model = Prophet(growth='logistic', daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=periods)
    future['cap'] = cap
    future['floor'] = floor
    prophet_forecast = prophet_model.predict(future)

    if model_choice != 'Prophet Only':
        df_residuals = df_prophet.copy()
        df_residuals['residuals'] = df_prophet['y'] - prophet_forecast.loc[:len(df_prophet)-1, 'yhat']
        df_residuals_features = create_features(df_residuals)
        future_features = create_features(future.copy())
        features = ['dayofweek', 'month', 'year']
        X_train, y_train = df_residuals_features[features], df_residuals_features['residuals']
        X_future = future_features[features]
        
        if model_choice == 'Prophet + RandomForest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else: # Prophet + XGBoost
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42)
        
        model.fit(X_train, y_train)
        residual_prediction = model.predict(X_future)
        prophet_forecast['yhat'] += residual_prediction

    prophet_forecast['yhat'] = np.clip(prophet_forecast['yhat'], floor, cap)
    metrics = {'mae': mean_absolute_error(df_prophet['y'], prophet_forecast.loc[:len(df_prophet)-1, 'yhat']),
               'rmse': np.sqrt(mean_squared_error(df_prophet['y'], prophet_forecast.loc[:len(df_prophet)-1, 'yhat']))}
    
    return prophet_forecast[['ds', 'yhat']], metrics

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state(db)
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/2560px-McDonald%27s_Golden_Arches.svg.png")
        st.title(f"Welcome, *{st.session_state.name}*")
        st.markdown("---")
        
        model_choice = st.selectbox("Select Forecasting Model:", ("Prophet Only", "Prophet + RandomForest", "Prophet + XGBoost"), index=1)
        
        # --- ADD-ON SALES INPUT (RESTORED) ---
        st.session_state.add_on_sales = st.number_input(
            "Add-on Sales (₱)", 
            min_value=0.0, 
            value=st.session_state.add_on_sales,
            help="Add a fixed amount to each day's sales forecast for promotions or special events."
        )
        st.markdown("---")

        if len(st.session_state.historical_df) < 15:
            st.warning("Provide at least 15 days of data.")
            st.button("🔄 Generate Forecast", type="primary", use_container_width=True, disabled=True)
        else:
            if st.button("🔄 Generate Forecast", type="primary", use_container_width=True):
                with st.spinner("🧠 Building intelligent forecast models..."):
                    hist_df = calculate_atv(st.session_state.historical_df.copy())
                    last_hist_date = hist_df['date'].max()
                    today = pd.to_datetime('today').normalize()
                    periods = max((today + pd.Timedelta(days=15) - last_hist_date).days, 15)

                    cust_f, cust_m = train_and_forecast_component(hist_df, periods, 'customers', model_choice)
                    atv_f, atv_m = train_and_forecast_component(hist_df, periods, 'atv', model_choice)
                    
                    if not cust_f.empty and not atv_f.empty:
                        combo_f = pd.merge(cust_f.rename(columns={'yhat': 'forecast_customers'}), atv_f.rename(columns={'yhat': 'forecast_atv'}), on='ds')
                        combo_f['forecast_sales'] = (combo_f['forecast_customers'] * combo_f['forecast_atv']) + st.session_state.add_on_sales
                        st.session_state.forecast_df = combo_f
                        st.session_state.metrics = {'customers': cust_m, 'atv': atv_m}
                        st.success("Forecast generated!")
                    else:
                        st.error("Forecast generation failed.")

        st.markdown("---")
        st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)

    st.title("🍔 McDonald's AI Sales Forecaster")
    tabs = st.tabs(["🔮 Forecast Dashboard", "🗂️ Data Management"])
    
    with tabs[0]:
        if not st.session_state.forecast_df.empty:
            st.header("15-Day Component Forecast")
            today = pd.to_datetime('today').normalize()
            future_forecast_df = st.session_state.forecast_df[st.session_state.forecast_df['ds'] >= today].copy()

            if future_forecast_df.empty:
                st.warning("No future dates in the forecast to display.")
            else:
                disp_cols = {'ds':'Date', 'forecast_customers':'Predicted Customers', 'forecast_atv':'Predicted Avg Sale (₱)', 'forecast_sales':'Predicted Sales (₱)'}
                display_df = future_forecast_df.rename(columns=disp_cols)
                final_cols_order = [v for k,v in disp_cols.items() if v in display_df.columns]
                st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}', 'Predicted Avg Sale (₱)':'₱{:,.2f}', 'Predicted Sales (₱)':'₱{:,.2f}'}), use_container_width=True, height=560)
        else:
            st.info("Click the 'Generate Forecast' button to begin.")

    with tabs[1]:
        st.header("Manage Your Data")
        with st.expander("➕ Add New Daily Record"):
            with st.form("new_record_form", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                with c1: new_date = st.date_input("Date")
                with c2: new_sales = st.number_input("Total Sales (₱)", min_value=0.0, format="%.2f")
                with c3: new_customers = st.number_input("Customer Count", min_value=0)
                if st.form_submit_button("💾 Save Record"):
                    add_to_firestore(db, 'historical_data', {"date": new_date, "sales": new_sales, "customers": new_customers})
                    st.session_state.historical_df = load_from_firestore(db, 'historical_data')
                    st.success("Record added!")
                    st.rerun()
        
        # --- CSV UPLOADER (RESTORED) ---
        with st.expander("📤 Upload Historical Data from CSV"):
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    upload_df = pd.read_csv(uploaded_file)
                    required_cols = {'date', 'sales', 'customers'}
                    if not required_cols.issubset(upload_df.columns):
                        st.error(f"CSV must contain the following columns: {', '.join(required_cols)}")
                    else:
                        st.write("Preview of data to be uploaded:")
                        st.dataframe(upload_df.head())
                        if st.button("Migrate this data to Firestore"):
                            with st.spinner("Uploading data..."):
                                for i, row in upload_df.iterrows():
                                    add_to_firestore(db, 'historical_data', row.to_dict())
                            st.success("Data migration complete!")
                            st.session_state.historical_df = load_from_firestore(db, 'historical_data')
                            st.rerun()
                except Exception as e:
                    st.error(f"An error occurred while processing the file: {e}")

        st.subheader("Edit Historical Data")
        df = st.session_state.historical_df.copy()
        if not df.empty:
            df['month_year'] = df['date'].dt.to_period('M').astype(str)
            month_options = sorted(df['month_year'].unique(), reverse=True)
            selected_month = st.selectbox("Select month to view/edit:", month_options)
            
            monthly_df = df[df['month_year'] == selected_month].copy()
            
            edited_df = st.data_editor(monthly_df, use_container_width=True, hide_index=True, key=f"editor_{selected_month}",
                column_config={"id": None, "month_year": None, "date": st.column_config.DateColumn("Date", disabled=True),
                               "sales": st.column_config.NumberColumn("Sales (₱)", format="₱%.2f"),
                               "customers": st.column_config.NumberColumn("Customers"), "atv": None}, disabled=['atv'])

            if st.button("💾 Save Changes", key=f"save_{selected_month}"):
                with st.spinner("Saving changes..."):
                    for i in edited_df.index:
                        if not monthly_df.loc[i].equals(edited_df.loc[i]):
                            doc_id = edited_df.loc[i, 'id']
                            update_data = edited_df.loc[i].drop(['id', 'month_year', 'atv', 'date']).to_dict()
                            update_in_firestore(db, 'historical_data', doc_id, update_data)
                st.success("Changes saved!")
                st.session_state.historical_df = load_from_firestore(db, 'historical_data')
                st.rerun()
        else:
            st.info("No historical data found. Add a record or upload a CSV to get started.")
