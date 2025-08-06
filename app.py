# app.py
import streamlit as st
import pandas as pd
import time
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objects as go

# --- Import from our modules ---
from data_processing import load_from_firestore
from forecasting import generate_forecast

# --- Page Configuration and Styling ---
st.set_page_config(page_title="AI Forecaster v5.5 Final", layout="wide")

# --- Firestore Initialization (This is now stable and correct) ---
@st.cache_resource
def init_firestore():
    try:
        if "firebase_credentials" not in st.secrets:
            st.error("`firebase_credentials` not found in Streamlit Secrets.")
            return None
        creds_dict = dict(st.secrets["firebase_credentials"])
        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: {type(e).__name__} - {e}")
        return None

# --- UI Rendering and Data Handling Functions ---

def render_historical_record(row, db_client):
    """Renders an editable historical data record using st.expander."""
    if 'doc_id' not in row or pd.isna(row['doc_id']):
        return

    date_str = row['date'].strftime('%B %d, %Y')
    expander_title = f"{date_str} - Sales: ₱{row.get('sales', 0):,.2f}, Customers: {row.get('customers', 0)}"
    
    with st.expander(expander_title):
        day_type = row.get('day_type', 'Normal Day')
        st.write(f"**Day Type:** {day_type}")
        
        with st.form(key=f"edit_form_{row['doc_id']}", border=False):
            day_type_options = ["Normal Day", "Not Normal Day"]
            try:
                current_index = day_type_options.index(day_type)
            except ValueError:
                current_index = 0
            
            updated_day_type = st.selectbox(
                "Correct Day Type",
                day_type_options,
                index=current_index,
                key=f"day_type_select_{row['doc_id']}"
            )
            
            if st.form_submit_button("💾 Update Record", use_container_width=True):
                update_data = {'day_type': updated_day_type}
                db_client.collection('historical_data').document(row['doc_id']).update(update_data)
                st.success(f"Record for {date_str} updated successfully!")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()

# --- Main Application ---

db = init_firestore()

if db:
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'model' not in st.session_state:
        st.session_state.model = None

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
        st.title("AI Forecaster v5.5")
        st.info("Robust Engine - Production Build")

        if st.button("🔄 Refresh Data & Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared. Rerunning...")
            time.sleep(1); st.rerun()

        # --- THIS IS THE CORRECTED AND FULLY IMPLEMENTED BUTTON LOGIC ---
        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            historical_df = load_from_firestore(db, 'historical_data')
            events_df = load_from_firestore(db, 'future_activities')

            if len(historical_df) < 90:
                st.error("Need at least 90 days of data for a reliable forecast.")
            else:
                with st.spinner("🧠 AI Engine is generating the forecast... (This may take a few minutes)"):
                    try:
                        forecast_df, model = generate_forecast(historical_df, events_df, periods=15)
                        st.session_state.forecast_df = forecast_df
                        st.session_state.model = model
                        st.success("Forecast Generated Successfully!")
                    except Exception as e:
                        st.error(f"An error occurred during forecast generation: {e}")

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
            fig.update_layout(title="Customer Forecast", yaxis_title="Predicted Customers")
            st.plotly_chart(fig, use_container_width=True)
            display_df = df[['ds', 'predicted_customers', 'predicted_atv', 'predicted_sales']].rename(columns={
                'ds': 'Date', 'predicted_customers': 'Predicted Customers',
                'predicted_atv': 'Predicted Avg Sale (₱)', 'predicted_sales': 'Predicted Sales (₱)'
            }).set_index('Date')
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Click 'Generate Forecast' in the sidebar to view your sales prediction.")

    with tabs[1]:
        st.header("💡 Forecast Insights")
        st.info("This shows the general importance of different features to the model's predictions.")
        if st.session_state.model:
            model = st.session_state.model
            try:
                val_dataloader = model.val_dataloader()
                if val_dataloader is not None:
                    importance = model.evaluate(val_dataloader, verbose=False)
                    fig = model.plot_variable_importances(importance)
                    st.pyplot(fig)
                else:
                    st.warning("Could not create validation dataloader for insights.")
            except Exception as e:
                st.error(f"Could not generate feature importance plot. Details: {e}")
        else:
            st.info("Generate a forecast to see the key drivers of customer behavior.")

    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        st.info("Here you can correct the 'Day Type' for past dates. This improves future forecasts.")
        @st.cache_data
        def get_historical_data_for_editing(_db_client):
            return load_from_firestore(_db_client, 'historical_data')

        historical_df_edit = get_historical_data_for_editing(db)
        if not historical_df_edit.empty:
            recent_df = historical_df_edit.sort_values(by="date", ascending=False).head(30)
            for _, row in recent_df.iterrows():
                render_historical_record(row, db)
        else:
            st.warning("No historical data found to display.")
else:
    st.warning("Could not connect to Firestore. Please check the detailed error message above and review your configuration.")
