import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objs as go
import yaml
from yaml.loader import SafeLoader
import io
import time
import os
import requests
from datetime import timedelta, date
import firebase_admin
from firebase_admin import credentials, firestore
import json
import logging

# --- Suppress Prophet's informational messages ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Forecaster",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom McDonald's Inspired CSS ---
def apply_custom_styling():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        /* --- Main Font & Colors --- */
        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }
        .main > div {
            background-color: #1a1a1a; /* Original dark background */
        }
        
        /* --- Clean Layout Adjustments --- */
        .block-container {
            padding-top: 2.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        /* --- Sidebar --- */
        [data-testid="stSidebar"] {
            background-color: #252525; /* Original sidebar color */
            border-right: 1px solid #444;
            width: 320px !important;
        }
        [data-testid="stSidebar-resize-handler"] {
            display: none;
        }
        
        /* --- Primary & Secondary Buttons --- */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
            border: none;
            padding: 10px 16px;
        }
        /* Primary Action Button Style (e.g., Generate Forecast, Save) */
        .stButton:has(button:contains("Generate")),
        .stButton:has(button:contains("Save")) > button {
            background: linear-gradient(45deg, #c8102e, #e01a37); /* Red gradient */
            color: #FFFFFF;
        }
        .stButton:has(button:contains("Generate")):hover > button,
        .stButton:has(button:contains("Save")):hover > button {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px 0 rgba(200, 16, 46, 0.4);
        }
        /* Secondary Action Button (e.g., Refresh, View All) */
        .stButton:has(button:contains("Refresh")),
        .stButton:has(button:contains("View All")),
        .stButton:has(button:contains("Back to Overview")) > button {
            border: 2px solid #c8102e;
            background: transparent;
            color: #c8102e;
        }
        .stButton:has(button:contains("Refresh")):hover > button,
        .stButton:has(button:contains("View All")):hover > button,
        .stButton:has(button:contains("Back to Overview")):hover > button {
            background: #c8102e;
            color: #ffffff;
        }

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            background-color: transparent;
            color: #d3d3d3;
            padding: 8px 14px; /* Compact padding */
            font-weight: 600;
            font-size: 0.9rem; /* Smaller font for tabs */
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #c8102e;
            color: #ffffff;
        }

        /* --- Expanders for Editing --- */
        .st-expander {
            border: 1px solid #444 !important;
            box-shadow: none;
            border-radius: 10px;
            background-color: #252525;
            margin-bottom: 0.5rem;
        }
        .st-expander header {
            font-size: 0.9rem;
            font-weight: 600;
            color: #d3d3d3;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization ---
@st.cache_resource
def init_firestore():
    try:
        if not firebase_admin._apps:
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
        st.error(f"Firestore Connection Error: Failed to initialize Firebase. Please check your Streamlit Secrets configuration. Error details: {e}")
        return None

# --- App State Management ---
def initialize_state_firestore(db_client):
    if 'db_client' not in st.session_state: st.session_state.db_client = db_client
    if 'historical_df' not in st.session_state: st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if 'events_df' not in st.session_state: st.session_state.events_df = load_from_firestore(db_client, 'future_activities')
    defaults = {
        'forecast_df': pd.DataFrame(), 
        'metrics': {}, 
        'name': "Store 688", 
        'authentication_status': True, 
        'forecast_components': pd.DataFrame(), 
        'migration_done': False,
        'show_recent_entries': False,
        'show_all_activities': False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- Data Processing and Feature Engineering ---
@st.cache_data(ttl="1h") # Unified cache time
def load_from_firestore(_db_client, collection_name):
    if _db_client is None: return pd.DataFrame()
    
    docs = _db_client.collection(collection_name).stream()
    records = []
    for doc in docs:
        record = doc.to_dict()
        record['doc_id'] = doc.id
        records.append(record)
        
    if not records: return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = df['date'].dt.tz_localize(None)
        df.dropna(subset=['date'], inplace=True)
    
    numeric_cols = ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers', 'potential_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if not df.empty:
        df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
        
    return df

def remove_outliers_iqr(df, column='sales'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    original_rows = len(df)
    cleaned_df = df[df[column] <= upper_bound].copy()
    removed_rows = original_rows - len(cleaned_df)
    return cleaned_df, removed_rows, upper_bound

def calculate_atv(df):
    sales = pd.to_numeric(df['sales'], errors='coerce').fillna(0); customers = pd.to_numeric(df['customers'], errors='coerce').fillna(0)
    with np.errstate(divide='ignore', invalid='ignore'): atv = np.divide(sales, customers)
    df['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df

@st.cache_data(ttl=3600)
def get_weather_forecast(days=16):
    try:
        url="https://api.open-meteo.com/v1/forecast";params={"latitude":10.48,"longitude":123.42,"daily":"weather_code,temperature_2m_max,precipitation_sum,wind_speed_10m_max","timezone":"Asia/Manila","forecast_days":days}
        response=requests.get(url,params=params);response.raise_for_status();data=response.json();df=pd.DataFrame(data['daily'])
        df.rename(columns={'time':'date','temperature_2m_max':'temp_max','precipitation_sum':'precipitation','wind_speed_10m_max':'wind_speed'},inplace=True)
        df['date']=pd.to_datetime(df['date']);df['weather']=df['weather_code'].apply(map_weather_code)
        return df
    except requests.exceptions.RequestException:return None

def map_weather_code(code):
    if code in[0,1]:return"Sunny"
    if code in[2,3]:return"Cloudy"
    if code in[51,53,55,61,63,65,80,81,82]:return"Rainy"
    if code in[95,96,99]:return"Storm"
    return"Cloudy"

def generate_recurring_local_events(start_date,end_date):
    local_events=[];current_date=start_date
    while current_date<=end_date:
        if current_date.day in[15,30]:local_events.append({'holiday':'Payday','ds':current_date,'lower_window':0,'upper_window':1});[local_events.append({'holiday':'Near_Payday','ds':current_date-timedelta(days=i),'lower_window':0,'upper_window':0})for i in range(1,3)]
        if current_date.month==7 and current_date.day==1:local_events.append({'holiday':'San Carlos Charter Day','ds':current_date,'lower_window':0,'upper_window':0})
        current_date+=timedelta(days=1)
    return pd.DataFrame(local_events)

# --- Core Forecasting Model ---
@st.cache_resource
def train_and_forecast_component(historical_df, events_df, periods, target_col):
    df_train = historical_df.copy()
    df_train.dropna(subset=['date', target_col], inplace=True)
    
    if df_train.empty or len(df_train) < 15:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    df_prophet = df_train.rename(columns={'date': 'ds', target_col: 'y'})[['ds', 'y']]
    
    start_date = df_train['date'].min()
    end_date = df_train['date'].max() + timedelta(days=periods)
    recurring_events = generate_recurring_local_events(start_date, end_date)
    all_manual_events = pd.concat([events_df.rename(columns={'date':'ds', 'event_name':'holiday'}), recurring_events])
    
    use_yearly_seasonality = len(df_train) >= 365

    prophet_model = Prophet(
        growth='linear',
        holidays=all_manual_events,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=use_yearly_seasonality, 
        changepoint_prior_scale=0.01,
        changepoint_range=0.8
    )
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    
    future = prophet_model.make_future_dataframe(periods=periods)
    prophet_forecast = prophet_model.predict(future)

    potential_cols = ['ds', 'trend', 'holidays', 'weekly', 'yearly', 'daily', 'yhat']
    existing_cols = [col for col in potential_cols if col in prophet_forecast.columns]
    forecast_components = prophet_forecast[existing_cols]

    metrics = {'mae': mean_absolute_error(df_prophet['y'], prophet_forecast.loc[:len(df_prophet)-1, 'yhat']), 'rmse': np.sqrt(mean_squared_error(df_prophet['y'], prophet_forecast.loc[:len(df_prophet)-1, 'yhat']))}
    
    return prophet_forecast[['ds', 'yhat']], metrics, forecast_components, prophet_model.holidays

# --- Plotting Functions & Firestore Data I/O ---
def add_to_firestore(db_client, collection_name, data, historical_df):
    if db_client is None: return
    
    if 'date' in data and pd.notna(data['date']):
        current_date = pd.to_datetime(data['date'])
        last_year_date = current_date - timedelta(days=364)
        
        hist_copy = historical_df.copy()
        hist_copy['date_only'] = pd.to_datetime(hist_copy['date']).dt.date
        
        last_year_record = hist_copy[hist_copy['date_only'] == last_year_date.date()]
        
        if not last_year_record.empty:
            data['last_year_sales'] = last_year_record['sales'].iloc[0]
            data['last_year_customers'] = last_year_record['customers'].iloc[0]
        else:
            data['last_year_sales'] = 0.0
            data['last_year_customers'] = 0.0
            
        data['date'] = current_date.to_pydatetime()
    else: 
        return

    all_cols = ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers', 'weather']
    for col in all_cols:
        if col in data and data[col] is not None:
            if col not in ['weather']:
                 data[col] = float(pd.to_numeric(data[col], errors='coerce'))
        else:
            if col not in ['weather']:
                data[col] = 0.0
            else:
                data[col] = "N/A"

    db_client.collection(collection_name).add(data)


def update_historical_record_in_firestore(db_client, doc_id, data):
    if db_client is None: return
    db_client.collection('historical_data').document(doc_id).update(data)

def update_activity_in_firestore(db_client, doc_id, data):
    if db_client is None: return
    if 'potential_sales' in data:
        data['potential_sales'] = float(data['potential_sales'])
    db_client.collection('future_activities').document(doc_id).update(data)

def delete_from_firestore(db_client, collection_name, doc_id):
    if db_client is None: return
    db_client.collection(collection_name).document(doc_id).delete()

def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

def plot_full_comparison_chart(hist,fcst,metrics,target):
    fig=go.Figure();fig.add_trace(go.Scatter(x=hist['date'],y=hist[target],mode='lines+markers',name='Historical Actuals',line=dict(color='#3b82f6')));fig.add_trace(go.Scatter(x=fcst['ds'],y=fcst['yhat'],mode='lines',name='Forecast',line=dict(color='#ffc72c',dash='dash')));title_text=f"{target.replace('_',' ').title()} Forecast";y_axis_title=title_text+' (₱)'if'atv'in target or'sales'in target else title_text
    fig.update_layout(title=f'Full Diagnostic: {title_text} vs. Historical',xaxis_title='Date',yaxis_title=y_axis_title,legend=dict(x=0.01,y=0.99),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');fig.add_annotation(x=0.02,y=0.95,xref="paper",yref="paper",text=f"<b>Model Perf:</b><br>MAE:{metrics.get('mae',0):.2f}<br>RMSE:{metrics.get('rmse',0):.2f}",showarrow=False,font=dict(size=12,color="white"),align="left",bgcolor="rgba(0,0,0,0.5)");return fig

def plot_forecast_breakdown(components,selected_date,all_events):
    day_data=components[components['ds']==selected_date].iloc[0];event_on_day=all_events[all_events['ds']==selected_date]
    x_data = ['Baseline Trend'];y_data = [day_data.get('trend', 0)];measure_data = ["absolute"]
    if 'weekly' in day_data and pd.notna(day_data['weekly']):
        x_data.append('Day of Week Effect');y_data.append(day_data['weekly']);measure_data.append('relative')
    if 'daily' in day_data and pd.notna(day_data['daily']):
        x_data.append('Time of Day Effect');y_data.append(day_data['daily']);measure_data.append('relative')
    if 'yearly' in day_data and pd.notna(day_data['yearly']):
        x_data.append('Time of Year Effect');y_data.append(day_data['yearly']);measure_data.append('relative')
    if 'holidays' in day_data and pd.notna(day_data['holidays']):
        holiday_text='Holidays/Events'if event_on_day.empty else f"Event: {event_on_day['holiday'].iloc[0]}"
        x_data.append(holiday_text);y_data.append(day_data['holidays']);measure_data.append('relative')
    x_data.append('Final Forecast');y_data.append(day_data['yhat']);measure_data.append('total')
    fig=go.Figure(go.Waterfall(name="Breakdown",orientation="v",measure=measure_data,x=x_data,textposition="outside",text=[f"{v:,.0f}"for v in y_data],y=y_data,connector={"line":{"color":"rgb(63,63,63)"}},increasing={"marker":{"color":"#2ca02c"}},decreasing={"marker":{"color":"#d62728"}},totals={"marker":{"color":"#1f77b4"}}));fig.update_layout(title=f"Customer Forecast Breakdown for {selected_date.strftime('%A,%B %d')}",showlegend=False,paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');return fig,day_data

def generate_insight_summary(day_data,selected_date):
    effects={'Day of the Week':day_data.get('weekly', 0),'Time of Year':day_data.get('yearly', 0),'Holidays/Events':day_data.get('holidays', 0)}
    significant_effects={k:v for k,v in effects.items()if abs(v)>1}
    summary=f"The forecast for **{selected_date.strftime('%A,%B %d')}** starts with a baseline trend of **{day_data.get('trend', 0):.0f} customers**.\n\n"
    if not significant_effects:
        summary += f"The final forecast of **{day_data.get('yhat', 0):.0f} customers** is driven primarily by this trend."
        return summary
    pos_drivers={k:v for k,v in significant_effects.items()if v>0};neg_drivers={k:v for k,v in significant_effects.items()if v<0}
    if pos_drivers:biggest_pos_driver=max(pos_drivers,key=pos_drivers.get);summary+=f"📈 Main positive driver is **{biggest_pos_driver}**,adding an estimated **{pos_drivers[biggest_pos_driver]:.0f} customers**.\n"
    if neg_drivers:biggest_neg_driver=min(neg_drivers,key=neg_drivers.get);summary+=f"📉 Main negative driver is **{biggest_neg_driver}**,reducing by **{abs(neg_drivers[biggest_neg_driver]):.0f} customers**.\n"
    summary+=f"\nAfter all factors,the final forecast is **{day_data.get('yhat', 0):.0f} customers**.";return summary

def render_activity_card(row, db_client, view_type='compact_list'):
    doc_id = row['doc_id']
    
    if view_type == 'compact_list':
        # Compact, one-line summary for the main dashboard
        date_str = pd.to_datetime(row['date']).strftime('%b %d, %Y')
        summary_line = f"**{date_str}** | {row['activity_name']}"
        
        with st.expander(summary_line):
            status = row['remarks']
            if status == 'Confirmed': color = '#22C55E'
            elif status == 'Needs Follow-up': color = '#F59E0B'
            elif status == 'Tentative': color = '#38BDF8'
            else: color = '#EF4444'
            st.markdown(f"**Status:** <span style='color:{color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
            st.markdown(f"**Potential Sales:** ₱{row['potential_sales']:,.2f}")

            with st.form(key=f"compact_update_form_{doc_id}", border=False):
                status_options = ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"]
                current_status_index = status_options.index(status) if status in status_options else 0
                updated_sales = st.number_input("Sales (₱)", value=float(row['potential_sales']), format="%.2f", key=f"compact_sales_{doc_id}")
                updated_remarks = st.selectbox("Status", options=status_options, index=current_status_index, key=f"compact_remarks_{doc_id}")
                
                update_col, delete_col = st.columns(2)
                with update_col:
                    if st.form_submit_button("💾 Update", use_container_width=True):
                        update_data = {"potential_sales": updated_sales, "remarks": updated_remarks}
                        update_activity_in_firestore(db, doc_id, update_data)
                        st.success("Activity updated!")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                with delete_col:
                    if st.form_submit_button("🗑️ Delete", use_container_width=True):
                        delete_from_firestore(db, 'future_activities', doc_id)
                        st.warning("Activity deleted.")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
    else: # 'grid' view
        # Use st.container(border=True) to create a visual box for each activity.
        with st.container(border=True):
            activity_date_formatted = pd.to_datetime(row['date']).strftime('%A, %B %d, %Y')
            
            # Card Content
            st.markdown(f"**{row['activity_name']}**")
            st.markdown(f"<small>📅 {activity_date_formatted}</small>", unsafe_allow_html=True)
            st.markdown(f"💰 ₱{row['potential_sales']:,.2f}")
            status = row['remarks']
            if status == 'Confirmed': color = '#22C55E'
            elif status == 'Needs Follow-up': color = '#F59E0B'
            elif status == 'Tentative': color = '#38BDF8'
            else: color = '#EF4444'
            st.markdown(f"Status: <span style='color:{color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
            
            # Form inside an expander
            with st.expander("Edit / Manage"):
                status_options = ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"]
                current_status_index = status_options.index(status) if status in status_options else 0
                with st.form(key=f"full_update_form_{doc_id}", border=False):
                    updated_sales = st.number_input("Sales (₱)", value=float(row['potential_sales']), format="%.2f", key=f"full_sales_{doc_id}")
                    updated_remarks = st.selectbox("Status", options=status_options, index=current_status_index, key=f"full_remarks_{doc_id}")
                    
                    update_col, delete_col = st.columns(2)
                    with update_col:
                        if st.form_submit_button("💾 Update", use_container_width=True):
                            update_data = {"potential_sales": updated_sales, "remarks": updated_remarks}
                            update_activity_in_firestore(db, doc_id, update_data)
                            st.success("Activity updated!")
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()
                    with delete_col:
                        if st.form_submit_button("🗑️ Delete", use_container_width=True):
                            delete_from_firestore(db, 'future_activities', doc_id)
                            st.warning("Activity deleted.")
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)
    if st.session_state["authentication_status"]:
        with st.sidebar:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/2560px-McDonald%27s_Golden_Arches.svg.png");st.title(f"Welcome, *{st.session_state['name']}*");st.markdown("---")
            st.info("Forecasting with Prophet Model")

            if st.button("🔄 Refresh Data from Firestore"):
                st.cache_data.clear()
                st.success("Data cache cleared. Rerunning to get latest data.")
                time.sleep(1)
                st.rerun()

            if st.button("📈 Generate Forecast", use_container_width=True):
                if len(st.session_state.historical_df) < 20: 
                    st.error("Please provide at least 20 days of data for reliable forecasting.")
                else:
                    with st.spinner("🧠 Building forecast model..."):
                        base_df = st.session_state.historical_df.copy()
                        cleaned_df, removed_count, upper_bound = remove_outliers_iqr(base_df, column='sales')
                        
                        if removed_count > 0:
                            st.warning(f"Removed {removed_count} outlier day(s) with sales over ₱{upper_bound:,.2f}.")

                        hist_df_with_atv = calculate_atv(cleaned_df)
                        
                        ev_df = st.session_state.events_df.copy()
                        
                        cust_f, cust_m, cust_c, all_h = train_and_forecast_component(hist_df_with_atv, ev_df, 15, 'customers')
                        atv_f, atv_m, _, _ = train_and_forecast_component(hist_df_with_atv, ev_df, 15, 'atv')
                        
                        if not cust_f.empty and not atv_f.empty:
                            combo_f = pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}), atv_f.rename(columns={'yhat':'forecast_atv'}), on='ds')
                            combo_f['forecast_sales'] = combo_f['forecast_customers'] * combo_f['forecast_atv']
                            
                            with st.spinner("🛰️ Fetching live weather..."):
                                weather_df = get_weather_forecast()
                            if weather_df is not None:
                                combo_f = pd.merge(combo_f, weather_df[['date', 'weather']], left_on='ds', right_on='date', how='left').drop(columns=['date'])
                            else:
                                combo_f['weather'] = 'Not Available'
                                
                            st.session_state.forecast_df = combo_f
                            st.session_state.metrics = {'customers': cust_m, 'atv': atv_m}
                            st.session_state.forecast_components = cust_c
                            st.session_state.all_holidays = all_h
                            st.success(f"Forecast generated successfully!")
                        else:
                            st.error("Forecast generation failed.")

            st.markdown("---")
            st.download_button("📥 Download Forecast", convert_df_to_csv(st.session_state.forecast_df), "forecast_data.csv", "text/csv", use_container_width=True, disabled=st.session_state.forecast_df.empty)
            st.download_button("📥 Download Historical", convert_df_to_csv(st.session_state.historical_df), "historical_data.csv", "text/csv", use_container_width=True)
        
        tabs=st.tabs(["🔮 Forecast Dashboard","💡 Forecast Insights", "✍️ Add/Edit Data", "📅 Future Activities", "📜 Historical Data"])
        
        with tabs[0]:
            if not st.session_state.forecast_df.empty:
                today=pd.to_datetime('today').normalize();future_forecast_df=st.session_state.forecast_df[st.session_state.forecast_df['ds']>=today].copy()
                if future_forecast_df.empty:st.warning("Forecast contains no future dates.")
                else:
                    disp_cols={'ds':'Date','forecast_customers':'Predicted Customers','forecast_atv':'Predicted Avg Sale (₱)','forecast_sales':'Predicted Sales (₱)','weather':'Predicted Weather'}
                    existing_disp_cols={k:v for k,v in disp_cols.items()if k in future_forecast_df.columns};display_df=future_forecast_df.rename(columns=existing_disp_cols);final_cols_order=[v for k,v in disp_cols.items()if k in existing_disp_cols]
                    st.markdown("#### Forecasted Values");st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}),use_container_width=True,height=560)
                    st.markdown("#### Forecast Visualization");fig=go.Figure();fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#ffc72c')));fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_customers'],mode='lines+markers',name='Customer Forecast',yaxis='y2',line=dict(color='#c8102e')));fig.update_layout(title='15-Day Sales & Customer Forecast',xaxis_title='Date',yaxis=dict(title='Predicted Sales (₱)',color='#ffc72c'),yaxis2=dict(title='Predicted Customers',overlaying='y',side='right',color='#c8102e'),legend=dict(x=0.01,y=0.99,orientation='h'),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');st.plotly_chart(fig,use_container_width=True)
                with st.expander("🔬 View Full Forecast vs. Historical Data"):
                    st.info("This view shows how the component models performed against past data.");d_t1,d_t2=st.tabs(["Customer Analysis","Avg. Transaction Analysis"]);hist_atv=calculate_atv(st.session_state.historical_df.copy())
                    with d_t1:st.plotly_chart(plot_full_comparison_chart(hist_atv,st.session_state.forecast_df.rename(columns={'forecast_customers':'yhat'}),st.session_state.metrics.get('customers',{}),'customers'),use_container_width=True)
                    with d_t2:st.plotly_chart(plot_full_comparison_chart(hist_atv,st.session_state.forecast_df.rename(columns={'forecast_atv':'yhat'}),st.session_state.metrics.get('atv',{}),'atv'),use_container_width=True)
            else:st.info("Click the 'Generate Component Forecast' button to begin.")
        with tabs[1]:
            if'forecast_components'not in st.session_state or st.session_state.forecast_components.empty:st.info("Generate a forecast first to see the breakdown of its drivers.")
            else:
                future_components=st.session_state.forecast_components[st.session_state.forecast_components['ds']>=pd.to_datetime('today').normalize()].copy()
                if not future_components.empty:
                    future_components['date_str']=future_components['ds'].dt.strftime('%A,%B %d,%Y')
                    selected_date_str=st.selectbox("Select a day to analyze its forecast drivers:",options=future_components['date_str'])
                    selected_date=future_components[future_components['date_str']==selected_date_str]['ds'].iloc[0]
                    breakdown_fig,day_data=plot_forecast_breakdown(st.session_state.forecast_components,selected_date,st.session_state.all_holidays)
                    st.plotly_chart(breakdown_fig,use_container_width=True);st.markdown("---");st.subheader("Insight Summary");st.markdown(generate_insight_summary(day_data,selected_date))
                else:st.warning("No future dates available in the forecast components to analyze.")
        with tabs[2]:
            form_col, display_col = st.columns([2, 3], gap="large")

            with form_col:
                st.subheader("✍️ Add New Daily Record")
                with st.form("new_record_form",clear_on_submit=True, border=False):
                    new_date=st.date_input("Date", date.today())
                    new_sales=st.number_input("Total Sales (₱)",min_value=0.0,format="%.2f")
                    new_customers=st.number_input("Customer Count",min_value=0)
                    new_addons=st.number_input("Add-on Sales (₱)",min_value=0.0,format="%.2f")
                    new_weather=st.selectbox("Weather Condition",["Sunny","Cloudy","Rainy","Storm"],help="Describe general weather.")
                    if st.form_submit_button("✅ Save Record"):
                        new_rec={"date":new_date,"sales":new_sales,"customers":new_customers,"weather":new_weather,"add_on_sales":new_addons}
                        add_to_firestore(db,'historical_data',new_rec, st.session_state.historical_df)
                        st.cache_data.clear()
                        st.success("Record added to Firestore!");
                        time.sleep(1)
                        st.rerun()
            
            with display_col:
                if st.button("🗓️ Show/Hide Recent Entries"):
                    st.session_state.show_recent_entries = not st.session_state.show_recent_entries
                
                if st.session_state.show_recent_entries:
                    st.subheader("🗓️ Recent Entries")
                    recent_df = st.session_state.historical_df.copy().sort_values(by="date", ascending=False).head(10)
                    
                    if not recent_df.empty:
                        for _, row in recent_df.iterrows():
                            doc_id = row['doc_id']
                            date_str = pd.to_datetime(row['date']).strftime('%b %d, %Y')
                            summary_line = f"**{date_str}** | Sales: ₱{row.get('sales', 0):,.2f} | Customers: {row.get('customers', 0)} | Add-ons: ₱{row.get('add_on_sales', 0):,.2f}"
                            with st.expander(summary_line):
                                with st.form(key=f"update_hist_{doc_id}", border=False):
                                    st.markdown("##### Edit Record")
                                    cols = st.columns(3)
                                    updated_sales = cols[0].number_input("Total Sales (₱)", value=float(row.get('sales', 0)), format="%.2f")
                                    updated_customers = cols[1].number_input("Customers", value=int(row.get('customers', 0)))
                                    updated_addons = cols[2].number_input("Add-on Sales (₱)", value=float(row.get('add_on_sales', 0)), format="%.2f")
                                    updated_weather = st.selectbox("Weather", ["Sunny","Cloudy","Rainy","Storm"], index=["Sunny","Cloudy","Rainy","Storm"].index(row.get('weather', 'Sunny')))
                                    
                                    update_col, delete_col, _ = st.columns([1,1,3])
                                    if update_col.form_submit_button("💾 Update", use_container_width=True):
                                        update_data = {
                                            "sales": updated_sales,
                                            "customers": updated_customers,
                                            "add_on_sales": updated_addons,
                                            "weather": updated_weather
                                        }
                                        update_historical_record_in_firestore(db, doc_id, update_data)
                                        st.success("Record updated!")
                                        st.cache_data.clear()
                                        time.sleep(1)
                                        st.rerun()

                                    if delete_col.form_submit_button("🗑️ Delete", use_container_width=True):
                                        delete_from_firestore(db, 'historical_data', doc_id)
                                        st.warning("Record deleted.")
                                        st.cache_data.clear()
                                        time.sleep(1)
                                        st.rerun()
                    else:
                        st.info("No recent data to display.")
        
        with tabs[3]:
            def set_view_all(): st.session_state.show_all_activities = True
            def set_overview(): st.session_state.show_all_activities = False

            if st.session_state.get('show_all_activities'):
                # --- ALL ACTIVITIES VIEW ---
                st.markdown("#### All Upcoming Activities")
                st.button("⬅️ Back to Overview", on_click=set_overview)
                
                activities_df = load_from_firestore(db, 'future_activities')
                all_upcoming_df = activities_df[pd.to_datetime(activities_df['date']).dt.date >= date.today()].copy()
                
                if all_upcoming_df.empty:
                    st.info("No upcoming activities scheduled.")
                else:
                    all_upcoming_df['month_year'] = all_upcoming_df['date'].dt.strftime('%B %Y')
                    sorted_months_df = all_upcoming_df.sort_values('date')
                    month_tabs_list = sorted_months_df['month_year'].unique().tolist()
                    
                    if month_tabs_list:
                        month_tabs = st.tabs(month_tabs_list)
                        for i, tab in enumerate(month_tabs):
                            with tab:
                                month_name = month_tabs_list[i]
                                month_df = sorted_months_df[sorted_months_df['month_year'] == month_name]
                                
                                # --- Monthly Summary Metrics ---
                                header_cols = st.columns([2, 1, 1])
                                with header_cols[1]:
                                    total_sales = month_df['potential_sales'].sum()
                                    st.metric(label="Total Expected Sales", value=f"₱{total_sales:,.2f}")
                                with header_cols[2]:
                                    unconfirmed_count = len(month_df[month_df['remarks'] != 'Confirmed'])
                                    st.metric(label="Unconfirmed Activities", value=unconfirmed_count)
                                st.markdown("---")
                                # --- End Summary ---

                                activities = month_df.to_dict('records')
                                for i in range(0, len(activities), 4):
                                    cols = st.columns(4)
                                    row_activities = activities[i:i+4]
                                    for j, activity in enumerate(row_activities):
                                        with cols[j]:
                                            render_activity_card(activity, db, view_type='grid')

            else:
                # --- DEFAULT OVERVIEW ---
                col1, col2 = st.columns([1, 2], gap="large")

                with col1:
                    st.markdown("##### Add New Activity")
                    with st.form("new_activity_form", clear_on_submit=True, border=True):
                        activity_name = st.text_input("Activity/Event Name", placeholder="e.g., Catering for Birthday")
                        activity_date = st.date_input("Date of Activity", min_value=date.today())
                        potential_sales = st.number_input("Potential Sales (₱)", min_value=0.0, format="%.2f")
                        remarks = st.selectbox("Status", ["Confirmed", "Needs Follow-up", "Tentative", "Cancelled"])
                        
                        submitted = st.form_submit_button("✅ Save Activity", use_container_width=True)
                        if submitted:
                            if activity_name and activity_date:
                                new_activity = {
                                    "activity_name": activity_name,
                                    "date": pd.to_datetime(activity_date).to_pydatetime(),
                                    "potential_sales": float(potential_sales),
                                    "remarks": remarks
                                }
                                db.collection('future_activities').add(new_activity)
                                st.success(f"Activity '{activity_name}' saved!")
                                st.cache_data.clear()
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning("Activity name and date are required.")

                with col2:
                    st.markdown("##### Next 10 Upcoming Activities")
                    
                    btn_cols = st.columns(2)
                    btn_cols[0].button("🔄 Refresh List", key='refresh_activities', use_container_width=True, on_click=st.rerun)
                    btn_cols[1].button("📂 View All Upcoming Activities", use_container_width=True, on_click=set_view_all)
                    
                    st.markdown("---",)
                    activities_df = load_from_firestore(db, 'future_activities')
                    upcoming_df = activities_df[pd.to_datetime(activities_df['date']).dt.date >= date.today()].copy().head(10)
                    
                    if upcoming_df.empty:
                        st.info("No upcoming activities scheduled.")
                    else:
                        for _, row in upcoming_df.iterrows():
                            render_activity_card(row, db, view_type='compact_list')


        with tabs[4]:
            st.subheader("View Historical Data")
            df = st.session_state.historical_df.copy()
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.dropna(subset=['date'], inplace=True)

                all_years = sorted(df['date'].dt.year.unique(), reverse=True)

                if all_years:
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_year = st.selectbox("Select Year to View:", options=all_years)

                    df_year_filtered = df[df['date'].dt.year == selected_year]
                    
                    all_months = sorted(df_year_filtered['date'].dt.strftime('%B').unique(), key=lambda m: pd.to_datetime(m, format='%B').month, reverse=True)

                    if all_months:
                        with col2:
                            selected_month_str = st.selectbox("Select Month to View:", options=all_months)

                        selected_month_num = pd.to_datetime(selected_month_str, format='%B').month

                        filtered_df = df[(df['date'].dt.year == selected_year) & (df['date'].dt.month == selected_month_num)].copy().reset_index(drop=True)

                        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                    else:
                        st.write(f"No data available for the year {selected_year}.")
                else:
                    st.write("No historical data to display.")
            else:
                st.write("No historical data to display.")
