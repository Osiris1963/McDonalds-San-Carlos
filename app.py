import streamlit as st
import pandas as pd
import time
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objects as go

# --- Import from our new, re-engineered modules ---
from data_processing import load_from_firestore
from forecasting import generate_forecast

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Unified AI Forecaster v4.0",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a professional look ---
def apply_custom_styling():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
        .main > div { background-color: #1a1a1a; color: #e0e0e0; }
        .block-container { padding: 2rem 2rem !important; }
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
        .stDataFrame { border: 1px solid #444; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization (Cached for performance) ---
@st.cache_resource
def init_firestore():
    """Initializes a connection to Firestore using Streamlit Secrets."""
    try:
        if not firebase_admin._apps:
            # Fallback for local development if secrets aren't set
            if 'firebase_credentials' not in st.secrets:
                 st.error("Firebase credentials not found in Streamlit Secrets.")
                 return None
            
            creds_dict = st.secrets.firebase_credentials.to_dict()
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: {e}")
        return None

# --- Data Loading (Cached to avoid frequent DB calls) ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_data(_db_client):
    """Loads all necessary data from Firestore."""
    if _db_client is None:
        return pd.DataFrame(), pd.DataFrame()
    historical_df = load_from_firestore(_db_client, 'historical_data')
    events_df = load_from_firestore(_db_client, 'future_activities')
    return historical_df, events_df

# --- Plotting Function ---
def create_forecast_plot(hist_df, fcst_df, target):
    """Creates a Plotly chart to visualize historical data and forecasts."""
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df[target],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#ffffff', width=2),
        marker=dict(size=4)
    ))

    # Add forecast trace
    fig.add_trace(go.Scatter(
        x=fcst_df['ds'],
        y=fcst_df[f'forecast_{target}'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#c8102e', width=3, dash='dash'),
        marker=dict(size=5)
    ))
    
    title_map = {'sales': 'Sales (PHP)', 'customers': 'Customer Count', 'atv': 'Average Transaction Value (PHP)'}
    
    fig.update_layout(
        title=f'Historical vs. Forecasted {title_map.get(target, target.capitalize())}',
        xaxis_title='Date',
        yaxis_title=title_map.get(target, target.capitalize()),
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Main Application ---
apply_custom_styling()
db = init_firestore()

if db:
    # Initialize session state
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("Unified AI Forecaster")
        st.info("Deep Learning Production Build v4.0")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.success("Cache cleared. Rerunning...")
            time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df, events_df = get_data(db)

            if len(historical_df) < 30: # Deep learning needs a reasonable amount of data
                st.error("Need at least 30 days of historical data for the model to train.")
            else:
                st.session_state.forecast_df = pd.DataFrame() # Clear previous forecast
                with st.spinner("🧠 Training Unified Deep Learning Model... This may take a minute."):
                    forecast_df = generate_forecast(historical_df, events_df, periods=15)
                
                if not forecast_df.empty:
                    st.session_state.forecast_df = forecast_df
                    st.success("Forecast Generated Successfully!")
                else:
                    st.error("Forecast generation failed. Check data or model logs.")

    # --- Main Content Tabs ---
    tab_list = ["📈 Forecast Dashboard", "🔢 Forecast Data", "✍️ Edit Historical Data"]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        st.header("📈 Forecast Dashboard")
        if not st.session_state.forecast_df.empty:
            historical_df, _ = get_data(db)
            
            # Create plots for all key metrics
            st.plotly_chart(create_forecast_plot(historical_df, st.session_state.forecast_df, 'sales'), use_container_width=True)
            st.plotly_chart(create_forecast_plot(historical_df, st.session_state.forecast_df, 'customers'), use_container_width=True)
            st.plotly_chart(create_forecast_plot(historical_df, st.session_state.forecast_df, 'atv'), use_container_width=True)
        else:
            st.info("Click 'Generate Forecast' in the sidebar to begin.")

    with tabs[1]:
        st.header("🔢 Forecast Data")
        if not st.session_state.forecast_df.empty:
            df_to_show = st.session_state.forecast_df.rename(columns={
                'ds': 'Date', 'forecast_customers': 'Predicted Customers',
                'forecast_atv': 'Predicted Avg. Sale (₱)', 'forecast_sales': 'Predicted Sales (₱)'
            }).set_index('Date')
            st.dataframe(df_to_show, use_container_width=True)
        else:
            st.info("Generate a forecast to see the raw data output.")

    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        st.info("Here you can flag past dates where unusual events occurred.")
        historical_df, _ = get_data(db)
        if not historical_df.empty:
            # Display recent records for editing
            recent_df = historical_df.sort_values(by="date", ascending=False).head(30)
            for _, row in recent_df.iterrows():
                with st.expander(f"{row['date'].strftime('%B %d, %Y')} - Sales: ₱{row.get('sales', 0):,.2f}"):
                    st.write(f"**Customers:** {row.get('customers', 0)}")
                    st.write(f"**Day Type:** {row.get('day_type', 'Normal Day')}")
                    
                    # Form to update the day type
                    with st.form(key=f"edit_form_{row['doc_id']}", border=False):
                        day_type_options = ["Normal Day", "Not Normal Day"]
                        current_index = day_type_options.index(row.get('day_type', 'Normal Day'))
                        new_day_type = st.selectbox("Set Day Type", day_type_options, index=current_index, key=f"dtype_{row['doc_id']}")
                        
                        if st.form_submit_button("💾 Update", use_container_width=True):
                            doc_ref = db.collection('historical_data').document(row['doc_id'])
                            doc_ref.update({'day_type': new_day_type})
                            st.success(f"Updated record for {row['date'].strftime('%Y-%m-%d')}")
                            st.cache_data.clear() # Clear cache to reflect changes
                            time.sleep(1); st.rerun()
        else:
            st.info("No historical data found in the database.")
else:
    st.error("Fatal Error: Could not connect to Firestore. Please check your Streamlit Secrets configuration and network.")

