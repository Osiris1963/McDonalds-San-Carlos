import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objs as go
import yaml
from yaml.loader import SafeLoader
import io
import time
import os
import requests
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="McDonald's AI Forecaster",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom McDonald's Inspired CSS ---
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
        .st-emotion-cache-p5msec { border: 1px solid #444; border-radius: 10px; background-color: #2a2a2a; }
        .st-emotion-cache-p5msec .st-emotion-cache-17x134l,
        .st-emotion-cache-1r4qj8v, .st-emotion-cache-1xw8zdv { color: #ffc72c; }
    </style>""", unsafe_allow_html=True)

# --- Firestore Initialization ---
@st.cache_resource
def init_firestore():
    try:
        if not firebase_admin._apps:
            creds_json_str = st.secrets.firebase_credentials.key
            creds_dict = json.loads(creds_json_str)
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: Failed to parse credentials. Please ensure your JSON key is correctly formatted in Streamlit Secrets. Error details: {e}")
        return None

# --- App State Management ---
def initialize_state_firestore(db_client):
    if 'db_client' not in st.session_state: st.session_state.db_client = db_client
    if 'historical_df' not in st.session_state: st.session_state.historical_df = load_from_firestore(db_client, 'historical_data')
    if 'events_df' not in st.session_state: st.session_state.events_df = load_from_firestore(db_client, 'future_events')
    defaults = {'forecast_df': pd.DataFrame(), 'metrics': {}, 'name': "Store 688", 'authentication_status': True, 'sales_cap': 1000000.0, 'customers_cap': 2500, 'atv_cap': 500.0, 'forecast_components': pd.DataFrame(), 'migration_done': False}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- Firestore Data Operations ---
def load_from_firestore(db_client, collection_name):
    if db_client is None: return pd.DataFrame()
    docs = db_client.collection(collection_name).stream(); records = [doc.to_dict() for doc in docs]
    if not records: return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # --- DEFINITIVE TIMEZONE FIX ---
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = df['date'].dt.tz_localize(None)
        df.dropna(subset=['date'], inplace=True)
    
    existing_numeric_cols = [col for col in ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers'] if col in df.columns]
    for col in existing_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=existing_numeric_cols, inplace=True)
    
    for col in existing_numeric_cols:
         df[col] = df[col].astype(float)
    
    if not df.empty:
        df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    return df

def add_to_firestore(db_client, collection_name, data):
    if db_client is None: return
    if 'date' in data and pd.notna(data['date']):
        data['date'] = pd.to_datetime(data['date']).to_pydatetime()
    else: return
    for col in ['sales', 'customers', 'add_on_sales', 'last_year_sales', 'last_year_customers']:
        if col in data and data[col] is not None: data[col] = float(pd.to_numeric(data[col], errors='coerce'))
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

# --- Hyperlocal & Business Logic Features ---
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
def calculate_atv(df):
    sales = pd.to_numeric(df['sales'], errors='coerce').fillna(0); customers = pd.to_numeric(df['customers'], errors='coerce').fillna(0)
    with np.errstate(divide='ignore', invalid='ignore'): atv = np.divide(sales, customers)
    df['atv'] = np.nan_to_num(atv, nan=0.0, posinf=0.0, neginf=0.0)
    return df
@st.cache_resource
def train_and_forecast_component(historical_df,events_df,weather_df,periods,target_col,use_rf_correction,floor=0):
    df_train=historical_df.copy();df_train[target_col]=pd.to_numeric(df_train[target_col],errors='coerce');df_train.replace([np.inf,-np.inf],np.nan,inplace=True);df_train.dropna(subset=['date',target_col],inplace=True)
    if df_train.empty or len(df_train)<15:return pd.DataFrame(),{},pd.DataFrame(),pd.DataFrame()
    cap = df_train[target_col].max() * 1.25
    df_train['cap']=cap;df_train['floor']=floor;df_prophet=df_train.rename(columns={'date':'ds',target_col:'y'})[['ds','y','cap','floor']]
    start_date=df_train['date'].min();end_date=df_train['date'].max()+timedelta(days=periods);recurring_events=generate_recurring_local_events(start_date,end_date);all_manual_events=pd.concat([events_df.rename(columns={'date':'ds','event_name':'holiday'}),recurring_events])
    prophet_model=Prophet(growth='logistic',holidays=all_manual_events,daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True);prophet_model.add_country_holidays(country_name='PH');prophet_model.fit(df_prophet)
    all_holidays_for_model=prophet_model.holidays;future=prophet_model.make_future_dataframe(periods=periods);future['cap']=cap;future['floor']=floor
    if weather_df is not None:future_weather=pd.merge(future,weather_df,left_on='ds',right_on='date',how='left').ffill().bfill();df_rf=pd.merge(df_train,weather_df,on='date',how='left')
    else:future_weather=future.copy();[future_weather.update({col:np.nan})for col in['temp_max','precipitation','wind_speed','weather']];df_rf=df_train.copy()
    forecast_in_sample=prophet_model.predict(df_prophet);df_rf['residuals']=df_prophet['y'].values-forecast_in_sample['yhat'].values
    if'weather'not in df_rf.columns:df_rf['weather']='Cloudy'
    df_rf=pd.get_dummies(df_rf,columns=['weather'],drop_first=True,dummy_na=True);features=[col for col in df_rf.columns if col.startswith('weather_')or col in['add_on_sales','temp_max','precipitation','wind_speed']]
    for col in features:
        if col in df_rf.columns:df_rf[col]=pd.to_numeric(df_rf[col],errors='coerce')
    X=df_rf[features].fillna(df_rf[features].mean());y=df_rf['residuals']
    if use_rf_correction:
        rf_model=RandomForestRegressor(n_estimators=100,random_state=42,min_samples_split=5);rf_model.fit(X,y);future_rf_data=pd.get_dummies(future_weather,columns=['weather'],dummy_na=True)
        for col in X.columns:
            if col not in future_rf_data.columns:future_rf_data[col]=0
        future_rf_data['add_on_sales']=0;future_rf_data=future_rf_data[X.columns].fillna(df_rf[features].mean());future_residuals=rf_model.predict(future_rf_data);prophet_forecast=prophet_model.predict(future);prophet_forecast['yhat']+=future_residuals
    else:prophet_forecast=prophet_model.predict(future)
    prophet_forecast['yhat']=np.clip(prophet_forecast['yhat'],floor,cap);forecast_components=prophet_forecast[['ds','trend','holidays','weekly','yearly','yhat']];metrics={'mae':mean_absolute_error(df_prophet['y'],prophet_forecast.loc[:len(df_prophet)-1,'yhat']),'rmse':np.sqrt(mean_squared_error(df_prophet['y'],prophet_forecast.loc[:len(df_prophet)-1,'yhat']))}
    return prophet_forecast[['ds','yhat']],metrics,forecast_components,all_holidays_for_model
def plot_full_comparison_chart(hist,fcst,metrics,target):
    fig=go.Figure();fig.add_trace(go.Scatter(x=hist['date'],y=hist[target],mode='lines+markers',name='Historical Actuals',line=dict(color='#3b82f6')));fig.add_trace(go.Scatter(x=fcst['ds'],y=fcst['yhat'],mode='lines',name='Forecast',line=dict(color='#ffc72c',dash='dash')));title_text=f"{target.replace('_',' ').title()} Forecast";y_axis_title=title_text+' (₱)'if'atv'in target or'sales'in target else title_text
    fig.update_layout(title=f'Full Diagnostic: {title_text} vs. Historical',xaxis_title='Date',yaxis_title=y_axis_title,legend=dict(x=0.01,y=0.99),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');fig.add_annotation(x=0.02,y=0.95,xref="paper",yref="paper",text=f"<b>Model Perf:</b><br>MAE:{metrics.get('mae',0):.2f}<br>RMSE:{metrics.get('rmse',0):.2f}",showarrow=False,font=dict(size=12,color="white"),align="left",bgcolor="rgba(0,0,0,0.5)");return fig
def plot_forecast_breakdown(components,selected_date,all_events):
    day_data=components[components['ds']==selected_date].iloc[0];event_on_day=all_events[all_events['ds']==selected_date]
    holiday_text='Holidays/Events'if event_on_day.empty else f"Event: {event_on_day['holiday'].iloc[0]}";x_data=['Baseline Trend','Day of Week Effect','Seasonal Effect',holiday_text,'Final Forecast'];y_data=[day_data['trend'],day_data['weekly'],day_data['yearly'],day_data['holidays'],day_data['yhat']]
    fig=go.Figure(go.Waterfall(name="Breakdown",orientation="v",measure=["absolute","relative","relative","relative","total"],x=x_data,textposition="outside",text=[f"{v:,.0f}"for v in y_data],y=[y_data[0],y_data[1],y_data[2],y_data[3],y_data[4]],connector={"line":{"color":"rgb(63,63,63)"}},increasing={"marker":{"color":"#2ca02c"}},decreasing={"marker":{"color":"#d62728"}},totals={"marker":{"color":"#1f77b4"}}));fig.update_layout(title=f"Customer Forecast Breakdown for {selected_date.strftime('%A,%B %d')}",showlegend=False,paper_bgcolor='#1e1e1e',plot_bgcolor='#2a2a2a',font_color='white');return fig,day_data
def generate_insight_summary(day_data,selected_date):
    effects={'Day of the Week':day_data['weekly'],'Time of Year':day_data['yearly'],'Holidays/Events':day_data['holidays']};significant_effects={k:v for k,v in effects.items()if abs(v)>1}
    if not significant_effects:return f"For **{selected_date.strftime('%A,%B %d')}**,the forecast of **{day_data['yhat']:.0f} customers** is driven primarily by the baseline trend of **{day_data['trend']:.0f}**."
    pos_drivers={k:v for k,v in significant_effects.items()if v>0};neg_drivers={k:v for k,v in significant_effects.items()if v<0};summary=f"The forecast for **{selected_date.strftime('%A,%B %d')}** starts with a baseline trend of **{day_data['trend']:.0f} customers**.\n\n"
    if pos_drivers:biggest_pos_driver=max(pos_drivers,key=pos_drivers.get);summary+=f"📈 Main positive driver is **{biggest_pos_driver}**,adding an estimated **{pos_drivers[biggest_pos_driver]:.0f} customers**.\n"
    if neg_drivers:biggest_neg_driver=min(neg_drivers,key=neg_drivers.get);summary+=f"📉 Main negative driver is **{biggest_neg_driver}**,reducing by **{abs(neg_drivers[biggest_neg_driver]):.0f} customers**.\n"
    summary+=f"\nAfter all factors,the final forecast is **{day_data['yhat']:.0f} customers**.";return summary

# --- Main Application UI ---
apply_custom_styling()
db = init_firestore()
if db:
    initialize_state_firestore(db)
    if st.session_state["authentication_status"]:
        with st.sidebar:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/2560px-McDonald%27s_Golden_Arches.svg.png");st.title(f"Welcome, *{st.session_state['name']}*");st.markdown("---")
            st.info("The AI now dynamically analyzes your data to set realistic forecast ceilings.")
            if len(st.session_state.historical_df)<15:st.warning("Provide at least 15 days of data.");st.button("🔄 Generate Component Forecast",type="primary",use_container_width=True,disabled=True)
            else:
                if st.button("🔄 Generate Component Forecast",type="primary",use_container_width=True):
                    with st.spinner("🛰️ Fetching live weather..."):weather_df=get_weather_forecast()
                    with st.spinner("🧠 Building component models..."):
                        hist_df=calculate_atv(st.session_state.historical_df.copy());ev_df=st.session_state.events_df.copy()
                        cust_f,cust_m,cust_c,all_h=train_and_forecast_component(hist_df,ev_df,weather_df,15,'customers',use_rf_correction=True)
                        atv_f,atv_m,_,_=train_and_forecast_component(hist_df,ev_df,weather_df,15,'atv',use_rf_correction=False)
                        if not cust_f.empty and not atv_f.empty:
                            combo_f=pd.merge(cust_f.rename(columns={'yhat':'forecast_customers'}),atv_f.rename(columns={'yhat':'forecast_atv'}),on='ds');combo_f['forecast_sales']=combo_f['forecast_customers']*combo_f['forecast_atv']
                            if weather_df is not None:combo_f=pd.merge(combo_f,weather_df[['date','weather']],left_on='ds',right_on='date',how='left').drop(columns=['date'])
                            else:combo_f['weather']='Not Available'
                            st.session_state.forecast_df=combo_f;st.session_state.metrics={'customers':cust_m,'atv':atv_m};st.session_state.forecast_components=cust_c;st.session_state.all_holidays=all_h;st.success("Forecast generated!")
                        else:st.error("Forecast generation failed.")
            st.markdown("---");st.download_button("📥 Download Forecast",convert_df_to_csv(st.session_state.forecast_df),"forecast_data.csv","text/csv",use_container_width=True,disabled=st.session_state.forecast_df.empty);st.download_button("📥 Download Historical",convert_df_to_csv(st.session_state.historical_df),"historical_data.csv","text/csv",use_container_width=True)
        
        st.title("🍔 McDonald's AI Sales Forecaster");tabs=st.tabs(["🔮 Forecast Dashboard","💡 Forecast Insights","🗂️ Data Management"])
        with tabs[0]:
            if not st.session_state.forecast_df.empty:
                st.header("15-Day Component Forecast");today=pd.to_datetime('today').normalize();future_forecast_df=st.session_state.forecast_df[st.session_state.forecast_df['ds']>=today].copy()
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
            st.header("The 'Why' Engine: Understanding Your Forecast");
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
            st.header("Manage Your Data")
            with st.container(border=True):
                st.subheader("Database Migration Tool")
                if st.session_state.get('migration_done',False)or not load_from_firestore(db,'historical_data').empty:st.success("Data source is now Firestore. CSV files are no longer used.")
                else:
                    st.warning("Action Required: Your data is still in CSV files. Migrate it to the new Firestore database.");
                    if st.button("Migrate CSV Data to Firestore"):
                        with st.spinner("Migrating data... This may take a moment."):
                            try:
                                for file_name, collection_name in [('sample_historical_data.csv', 'historical_data'), ('sample_future_events.csv', 'future_events')]:
                                    df_csv = pd.read_csv(file_name)
                                    df_csv['date'] = pd.to_datetime(df_csv['date'], errors='coerce')
                                    df_csv.dropna(subset=['date'], inplace=True)
                                    for _, row in df_csv.iterrows():
                                        add_to_firestore(db, collection_name, row.to_dict())
                                    st.write(f"✅ Migrated {collection_name} successfully.")
                                st.session_state.migration_done = True;st.success("Migration complete!");st.rerun()
                            except Exception as e:
                                st.error(f"Migration Failed: {e}")
            st.info("The model automatically incorporates PH National Holidays, paydays, and San Carlos Charter Day. Manage other events below.")
            with st.expander("➕ Add New Daily Record",expanded=True):
                with st.form("new_record_form",clear_on_submit=True):
                    c1,c2,c3=st.columns(3);c4,c5=st.columns(2);
                    with c1:new_date=st.date_input("Date")
                    with c2:new_sales=st.number_input("Total Sales (₱)",min_value=0.0,format="%.2f")
                    with c3:new_customers=st.number_input("Customer Count",min_value=0)
                    with c4:new_weather=st.selectbox("Weather Condition",["Sunny","Cloudy","Rainy","Storm"],help="Describe general weather.")
                    with c5:new_addons=st.number_input("Add-on Sales (₱)",min_value=0.0,format="%.2f")
                    if st.form_submit_button("💾 Save Record"):
                        new_rec={"date":new_date,"sales":new_sales,"customers":new_customers,"weather":new_weather,"add_on_sales":new_addons}
                        add_to_firestore(db,'historical_data',new_rec);st.session_state.historical_df=load_from_firestore(db,'historical_data');st.success("Record added to Firestore!");st.rerun()
            with st.expander("📝 Manage Historical Data by Month"):
                df=st.session_state.historical_df.copy()
                if not df.empty and'date'in df.columns:
                    df['date']=pd.to_datetime(df['date'],errors='coerce');df.dropna(subset=['date'],inplace=True)
                    month_periods=df['date'].dt.to_period('M').unique();month_options=sorted([p.strftime('%B %Y')for p in month_periods],reverse=True)
                    if month_options:
                        selected_month_str=st.selectbox("Select Month",options=month_options);selected_period=pd.Period(selected_month_str);filtered_df=df[df['date'].dt.to_period('M')==selected_period].copy().reset_index(drop=True)
                        original_filtered_df=filtered_df.copy()
                        edited_df=st.data_editor(filtered_df,key=f"editor_{selected_month_str}",num_rows="dynamic",use_container_width=True,hide_index=True,column_config={"id":None,"sales":st.column_config.NumberColumn("Sales (₱)",format="₱%.2f"),"weather":"Weather Condition","add_on_sales":st.column_config.NumberColumn("Add-on Sales (₱)",format="₱%.2f")})
                        if not edited_df.equals(original_filtered_df):
                            for index,row in edited_df.iterrows():
                                if not row.equals(original_filtered_df.loc[index]):
                                    doc_id=row['id'];update_data=row.to_dict();del update_data['id']
                                    update_in_firestore(db,'historical_data',doc_id,update_data) # Send corrected data
                            st.session_state.historical_df=load_from_firestore(db,'historical_data');st.toast(f"Changes for {selected_month_str} saved!");time.sleep(1);st.rerun()
                    else:st.write("No historical data to display.")
                else:st.write("No historical data to display.")
