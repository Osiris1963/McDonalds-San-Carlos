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
st.set_page_config(page_title="AI Forecaster v5.4 Final", layout="wide")

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
                # Clear caches to force a data refresh on the next run
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
        st.title("AI Forecaster v5.4")
        st.info("Robust Engine - Production Build")

        if st.button("🔄 Refresh Data & Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared. Rerunning...")
            time.sleep(1); st.rerun()

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            # ... Forecast generation logic ...
            pass # This part is correct and unchanged

    tab_list = ["🔮 Forecast Dashboard", "💡 Forecast Insights", "✍️ Edit Data"]
    tabs = st.tabs(tab_list)

    with tabs[0]:
        # ... Dashboard tab logic is correct and unchanged ...
        pass
    
    with tabs[1]:
        # ... Insights tab logic is correct and unchanged ...
        pass

    # --- THIS IS THE CORRECTED "EDIT DATA" TAB ---
    with tabs[2]:
        st.header("✍️ Edit Historical Data")
        st.info("Here you can correct the 'Day Type' for past dates. This improves future forecasts by teaching the model about unusual days.")
        
        @st.cache_data
        def get_historical_data_for_editing(_db_client):
            """Cached function to fetch data specifically for the editing tab."""
            return load_from_firestore(_db_client, 'historical_data')

        # Fetch the data using the cached function
        historical_df_edit = get_historical_data_for_editing(db)
        
        if not historical_df_edit.empty:
            # Display the most recent 30 days for editing
            recent_df = historical_df_edit.sort_values(by="date", ascending=False).head(30)
            
            # Loop through the dataframe and render the editable record for each row
            for _, row in recent_df.iterrows():
                render_historical_record(row, db)
        else:
            st.warning("No historical data found to display.")
else:
    st.warning("Could not connect to Firestore. Please check the detailed error message above and review your configuration.")
