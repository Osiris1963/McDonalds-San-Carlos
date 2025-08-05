# app.py (Upgraded for TFT and Probabilistic Forecasting)
import streamlit as st
import pandas as pd
import time
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objs as go

# --- Import from our new, re-architected modules ---
from data_processing import load_from_firestore
from forecasting import generate_tft_forecast

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Sales Forecaster v4.0 (TFT)",
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
    </style>
    """, unsafe_allow_html=True)

# --- Firestore Initialization ---
@st.cache_resource
def init_firestore():
    """Initializes a connection to Firestore using Streamlit Secrets."""
    try:
        if not firebase_admin._apps:
            creds_dict = st.secrets.firebase_credentials
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: Failed to initialize. Check your Streamlit Secrets. Details: {e}")
        return None

def save_forecast_to_log(db_client, forecast_df):
    """Saves the generated quantile forecast to Firestore."""
    if db_client is None or forecast_df.empty:
        st.warning("Database client not available or forecast is empty. Skipping log.")
        return False

    try:
        batch = db_client.batch()
        log_collection_ref = db_client.collection('forecast_log_v2_tft')
        generated_on_ts = pd.to_datetime('today').normalize()

        for _, row in forecast_df.iterrows():
            doc_id = row['ds'].strftime('%Y-%m-%d')
            log_doc_ref = log_collection_ref.document(doc_id)
            log_data = {
                'generated_on': generated_on_ts,
                'forecast_for_date': row['ds'],
                'sales_p10': float(row['forecast_sales_p10']),
                'sales_p50': float(row['forecast_sales_p50']),
                'sales_p90': float(row['forecast_sales_p90']),
                'customers_p50': int(row['forecast_customers_p50']),
            }
            batch.set(log_doc_ref, log_data, merge=True)
        batch.commit()
        return True
    except Exception as e:
        st.error(f"Error logging forecast to database: {e}")
        return False

def render_historical_record(row, db_client):
    """Renders an editable historical data record."""
    # (This function remains unchanged from your original version)
    if 'doc_id' not in row or pd.isna(row['doc_id']):
        return
    date_str = row['date'].strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    with st.expander(expander_title):
        st.write(f"**Add-on Sales:** ₱{row.get('add_on_sales', 0):,.2f}")
        day_type = row.get('day_type', 'Normal Day')
        st.write(f"**Day Type:** {day_type}")
        if day_type == 'Not Normal Day':
            st.write(f"**Notes:** {row.get('day_type_notes', 'N/A')}")
        with st.form(key=f"edit_hist_{row['doc_id']}", border=False):
            st.markdown("**Edit Record**")
            updated_day_type = st.selectbox("Day Type", ["Normal Day", "Not Normal Day"], index=0 if day_type == 'Normal Day' else 1, key=f"day_type_{row['doc_id']}")
            if st.form_submit_button("💾 Update Day Type", use_container_width=True):
                db_client.collection('historical_data').document(row['doc_id']).update({'day_type': updated_day_type})
                st.success(f"Record for {date_str} updated!")
                st.cache_data.clear(); st.rerun()

def plot_forecast(df):
    """Generates a Plotly chart for the forecast with prediction intervals."""
    fig = go.Figure()
    # Upper bound
    fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast_sales_p90'], mode='lines', line=dict(width=0), name='Upper Bound', showlegend=False))
    # Lower bound, with fill to upper bound
    fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast_sales_p10'], mode='lines', line=dict(width=0), name='Lower Bound', fill='tonexty', fillcolor='rgba(200, 16, 46, 0.2)', showlegend=False))
    # Median forecast
    fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast_sales_p50'], mode='lines+markers', name='Median Forecast (Sales)', line=dict(color='#c8102e', width=3)))
    
    fig.update_layout(
        title="Sales Forecast with 80% Prediction Interval",
        xaxis_title="Date", yaxis_title="Predicted Sales (₱)",
        template="plotly_dark", font=dict(family="Poppins"),
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
    if 'tft_artifacts' not in st.session_state:
        st.session_state.tft_artifacts = {}

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v4.0")
        st.info("Temporal Fusion Transformer Build")

        if st.button("🔄 Refresh Data from Firestore"):
            st.cache_data.clear(); st.rerun()

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df = load_from_firestore(db, 'historical_data')
            events_df = load_from_firestore(db, 'future_activities')

            if len(historical_df) < 60:
                st.error("Need at least 60 days of data for the TFT model.")
            else:
                with st.spinner("🧠 Training Temporal Fusion Transformer... This may take a few minutes."):
                    forecast_df, tft_model, x, raw_preds = generate_tft_forecast(historical_df, events_df, periods=15)
                    st.session_state.forecast_df = forecast_df
                    st.session_state.tft_artifacts = {'model': tft_model, 'x': x, 'raw_preds': raw_preds}

                if not forecast_df.empty:
                    with st.spinner("📡 Logging forecast..."):
                        if save_forecast_to_log(db, forecast_df):
                            st.success("Forecast Generated and Logged!")
                        else:
                            st.warning("Forecast generated but failed to log.")
                else:
                    st.error("Forecast generation failed. Check data for unusual patterns.")

    tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "✍️ Edit Data"]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        st.header("🔮 Forecast Dashboard")
        if not st.session_state.forecast_df.empty:
            st.plotly_chart(plot_forecast(st.session_state.forecast_df), use_container_width=True)
            st.dataframe(st.session_state.forecast_df.set_index('ds'), use_container_width=True)
        else:
            st.info("Click 'Generate Forecast' in the sidebar to begin.")

    with tabs[1]:
        st.header("💡 Forecast Insights (TFT Interpretability)")
        if st.session_state.tft_artifacts.get('model'):
            st.info("Select a single forecast point to inspect what the model 'paid attention to' when making its prediction.")
            
            model = st.session_state.tft_artifacts['model']
            x = st.session_state.tft_artifacts['x']
            raw_preds = st.session_state.tft_artifacts['raw_preds']
            
            forecast_dates = st.session_state.forecast_df['ds'].dt.strftime('%Y-%m-%d').tolist()
            selected_date_idx = st.selectbox("Select a forecast date to inspect:", options=range(len(forecast_dates)), format_func=lambda i: forecast_dates[i])

            if selected_date_idx is not None:
                st.subheader(f"Interpretation for {forecast_dates[selected_date_idx]}")
                
                # Plot Prediction vs Actuals for context
                fig_pred = model.plot_prediction(x, raw_preds, idx=selected_date_idx, add_loss_to_title=True)
                fig_pred.suptitle(f"Prediction vs Actuals (Validation Set Context)")
                st.pyplot(fig_pred)

                # Plot Interpretation
                fig_interp = model.plot_interpretation(x, raw_preds, idx=selected_date_idx)
                st.pyplot(fig_interp)

        else:
            st.info("Generate a forecast to see the model's interpretation.")

    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        st.info("Correct the 'Day Type' for past dates if an unusual event occurred.")
        historical_df = load_from_firestore(db, 'historical_data')
        if not historical_df.empty:
            recent_df = historical_df.sort_values(by="date", ascending=False).head(30)
            for _, row in recent_df.iterrows():
                render_historical_record(row, db)
        else:
            st.info("No historical data found.")
else:
    st.error("Could not connect to Firestore. Please check your configuration and network.")
