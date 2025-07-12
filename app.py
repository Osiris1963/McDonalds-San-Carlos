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

# --- FIX: Suppress Prophet's informational messages ---
# This will hide the verbose output that appears in the sidebar during forecasting
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
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        /* --- Main Font & Colors --- */
        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }
        .main > div {
            background-color: #1a1a1a;
        }
        
        /* --- Clean Layout Adjustments --- */
        /* Adjust top padding to position the tabs lower */
        .block-container {
            padding-top: 3.5rem !important;
        }
        
        /* --- Sidebar --- */
        .st-emotion-cache-16txtl3 {
            background-color: #252525;
            border-right: 1px solid #444;
        }
        
        /* --- Lock Sidebar Width and Disable Resizing --- */
        [data-testid="stSidebar"] {
            width: 320px !important; /* Set a fixed width */
        }
        [data-testid="stSidebar-resize-handler"] {
            display: none; /* Hide the resize handle */
        }
        
        /* --- Primary Button (Generate Forecast) --- */
        .stButton > button {
            border: none;
            border-radius: 12px;
            color: #ffffff; /* White text on red button */
            font-weight: 600;
            background: linear-gradient(45deg, #c8102e, #e01a37); /* Red gradient */
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 15px 0 rgba(200, 16, 46, 0.4); /* Red glow */
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(200, 16, 46, 0.5);
        }
        
        /* Special case for Refresh button */
        .stButton:has(button:contains("Refresh Data")) > button {
             border: 2px solid #c8102e;
             background: transparent;
             color: #c8102e;
        }
        .stButton:has(button:contains("Refresh Data")) > button:hover {
            background: #c8102e;
            color: #ffffff;
        }


        /* --- Tabs --- */
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            background-color: transparent;
            color: #d3d3d3;
            padding: 10px 15px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #c8102e; /* Use classic red for selected tab */
            color: #ffffff;
            font-weight: 600;
        }

        /* --- Expanders as Cards --- */
        .st-expander {
            border: none !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            border-radius: 15px;
            background-color: #252525;
            margin-bottom: 1rem;
        }
        .st-expander header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
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
    if 'events_df' not in st.session_state: st.session_state.events_df = load_from_firestore(db_client, 'future_events')
    defaults = {'forecast_df': pd.DataFrame(), 'metrics': {}, 'name': "Store 688", 'authentication_status': True, 'forecast_components': pd.DataFrame(), 'migration_done': False}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- Data Processing and Feature Engineering ---
@st.cache_data(ttl="6h")
def load_from_firestore(_db_client, collection_name):
    if _db_client is None: return pd.DataFrame()
    docs = _db_client.collection(collection_name).stream(); records = [doc.to_dict() for doc in docs]
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = df['date'].dt.tz_localize(None)
        df.dropna(subset=['date'], inplace=True)
    existing_numeric_cols = [col for col in ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers'] if col in df.columns]
    for col in existing_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['sales', 'customers', 'add_on_sales'], inplace=True)
    for col in existing_numeric_cols:
         if col in df.columns: df[col] = df[col].astype(float)
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
    
    # --- FIX: Auto-populate last year's data ---
    if 'date' in data and pd.notna(data['date']):
        current_date = pd.to_datetime(data['date'])
        last_year_date = current_date - timedelta(days=364) # Use 364 for day-of-week alignment
        
        # Ensure historical_df['date'] is datetime
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        
        last_year_record = historical_df[historical_df['date'].dt.date == last_year_date.date()]
        
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


def update_in_firestore(db_client, collection_name, doc_id, data):
    if db_client is None: return
    if 'date' in data and pd.notna(data['date']):
        data['date'] = pd.to_datetime(data['date']).to_pydatetime()
    for col in ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers']:
        if col in data and data[col] is not None:
            data[col] = float(pd.to_numeric(data[col], errors='coerce'))
    db_client.collection(collection_name).document(doc_id).set(data)

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
    fig=go.Figure(go.Waterfall(name="Breakdown",orientation="v",measure=measure_data,x=x_data,textposition="outside",text=[f"{v:,.0f}"for v in y_data],y=y_data,connector={"line":{"color":"rgb(63,63,63)"}},increasing={"marker":{"color":"#2ca02c"}},decreasing={"marker":{"color":"#d62728"}},totals={"marker":{"color":"#1f77b4"}}));fig.update_layout(title=f"Customer Forecast Breakdown for {selected_date.strftime('%A,%B %d')}",showlegend=False,paper_bgcolor='#1e1e1e',plot_bgcolor='#2a2a2a',font_color='white');return fig,day_data

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
        
        tabs=st.tabs(["🔮 Forecast Dashboard","💡 Forecast Insights", "✍️ Add/Edit Data", "📜 Historical Data"])
        
        with tabs[0]:
            if not st.session_state.forecast_df.empty:
                today=pd.to_datetime('today').normalize();future_forecast_df=st.session_state.forecast_df[st.session_state.forecast_df['ds']>=today].copy()
                if future_forecast_df.empty:st.warning("Forecast contains no future dates.")
                else:
                    disp_cols={'ds':'Date','forecast_customers':'Predicted Customers','forecast_atv':'Predicted Avg Sale (₱)','forecast_sales':'Predicted Sales (₱)','weather':'Predicted Weather'}
                    existing_disp_cols={k:v for k,v in disp_cols.items()if k in future_forecast_df.columns};display_df=future_forecast_df.rename(columns=existing_disp_cols);final_cols_order=[v for k,v in disp_cols.items()if k in existing_disp_cols]
                    st.markdown("#### Forecasted Values");st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}),use_container_width=True,height=560)
                    st.markdown("#### Forecast Visualization");fig=go.Figure();fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#ffc72c')));fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_customers'],mode='lines+markers',name='Customer Forecast',yaxis='y2',line=dict(color='#c8102e')));fig.update_layout(title='15-Day Sales & Customer Forecast',xaxis_title='Date',yaxis=dict(title='Predicted Sales (₱)',color='#ffc72c'),yaxis2=dict(title='Predicted Customers',overlaying='y',side='right',color='#c8102e'),legend=dict(x=0.01,y=0.99,orientation='h'),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#1e1e1e',plot_bgcolor='#2a2a2a',font_color='white');st.plotly_chart(fig,use_container_width=True)
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
            form_col, display_col = st.columns([1, 1], gap="large")

            with form_col:
                st.subheader("✍️ Add New Daily Record")
                with st.form("new_record_form",clear_on_submit=True, border=False):
                    new_date=st.date_input("Date", date.today())
                    new_sales=st.number_input("Total Sales (₱)",min_value=0.0,format="%.2f")
                    new_customers=st.number_input("Customer Count",min_value=0)
                    new_addons=st.number_input("Add-on Sales (₱)",min_value=0.0,format="%.2f")
                    new_weather=st.selectbox("Weather Condition",["Sunny","Cloudy","Rainy","Storm"],help="Describe general weather.")
                    if st.form_submit_button("✅ Save Record", use_container_width=True):
                        new_rec={"date":new_date,"sales":new_sales,"customers":new_customers,"weather":new_weather,"add_on_sales":new_addons}
                        add_to_firestore(db,'historical_data',new_rec, st.session_state.historical_df)
                        st.cache_data.clear()
                        st.success("Record added to Firestore!");
                        time.sleep(1)
                        st.rerun()
            
            with display_col:
                st.subheader("🗓️ Recent Entries")
                recent_df = st.session_state.historical_df.copy()
                if not recent_df.empty:
                    recent_df['date'] = pd.to_datetime(recent_df['date']).dt.date
                    display_cols = ['date', 'sales', 'customers', 'last_year_sales', 'last_year_customers', 'weather']
                    # Ensure columns exist before trying to display them
                    cols_to_show = [col for col in display_cols if col in recent_df.columns]
                    
                    st.dataframe(
                        recent_df[cols_to_show].sort_values(by="date", ascending=False).head(7),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                            "sales": st.column_config.NumberColumn("Sales (₱)", format="₱%,.2f"),
                            "customers": st.column_config.NumberColumn("Customers", format="%d"),
                            "last_year_sales": st.column_config.NumberColumn("LY Sales (₱)", format="₱%,.2f"),
                            "last_year_customers": st.column_config.NumberColumn("LY Customers", format="%d"),
                            "weather": "Weather"
                        }
                    )
                else:
                    st.info("No recent data to display.")

        with tabs[3]:
            st.subheader("View Historical Data")
            df=st.session_state.historical_df.copy()
            if not df.empty and'date'in df.columns:
                df['date']=pd.to_datetime(df['date'],errors='coerce');df.dropna(subset=['date'],inplace=True)
                month_periods=df['date'].dt.to_period('M').unique();month_options=sorted([p.strftime('%B %Y')for p in month_periods],reverse=True)
                if month_options:
                    selected_month_str=st.selectbox("Select Month to View:",options=month_options);selected_period=pd.Period(selected_month_str);filtered_df=df[df['date'].dt.to_period('M')==selected_period].copy().reset_index(drop=True)
                    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                else:st.write("No historical data to display.")
            else:st.write("No historical data to display.")
