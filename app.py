# app.py - COGNITIVE FORECASTING ENGINE v12.0
# Complete Streamlit Dashboard with all pages implemented
# v12.0 Improvements: Bug fixes, performance optimization, Philippine holidays,
# proper CI, forecast decomposition, anomaly handling, and real confidence scoring

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import timedelta, datetime

from data_processing import (
    load_from_firestore, 
    create_advanced_features,
    calculate_contextual_window_features,
    calculate_recent_trend_features
)
from forecasting import (
    generate_full_forecast, 
    backtest_model,
    calculate_accuracy_metrics,
    EnsembleForecaster,
    TREND_AWARE_MODE
)

# Check for intelligent mode
try:
    from intelligent_engine import CognitiveForecaster
    INTELLIGENT_MODE = True
except ImportError as e:
    INTELLIGENT_MODE = False
    print(f"⚠️ Intelligent mode unavailable: {e}")

# Check for trend intelligence
try:
    from trend_intelligence import TrendAnalyzer, apply_trend_intelligence
    TREND_MODE = True
except ImportError as e:
    TREND_MODE = False
    print(f"⚠️ Trend intelligence unavailable: {e}")

# --- Page Configuration ---
st.set_page_config(
    page_title="Cognitive Forecaster v12.0", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .accuracy-good { color: #10B981; font-weight: bold; }
    .accuracy-medium { color: #F59E0B; font-weight: bold; }
    .accuracy-poor { color: #EF4444; font-weight: bold; }
    .trend-down { color: #EF4444; font-weight: bold; }
    .trend-up { color: #10B981; font-weight: bold; }
    .trend-neutral { color: #6B7280; }
    .confidence-high { background: #10B981; color: white; padding: 2px 8px; border-radius: 4px; }
    .confidence-medium { background: #F59E0B; color: white; padding: 2px 8px; border-radius: 4px; }
    .confidence-low { background: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; }
    .briefing-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .signal-down { 
        background: #FEE2E2; 
        border-left: 4px solid #EF4444; 
        padding: 10px; 
        margin: 5px 0;
        border-radius: 4px;
    }
    .signal-up { 
        background: #D1FAE5; 
        border-left: 4px solid #10B981; 
        padding: 10px; 
        margin: 5px 0;
        border-radius: 4px;
    }
    .signal-neutral { 
        background: #F3F4F6; 
        border-left: 4px solid #6B7280; 
        padding: 10px; 
        margin: 5px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# --- Firestore Initialization ---
@st.cache_resource
def init_db():
    if not firebase_admin._apps:
        creds = st.secrets.firebase_credentials.to_dict()
        creds['private_key'] = creds['private_key'].replace('\\n', '\n')
        firebase_admin.initialize_app(credentials.Certificate(creds))
    return firestore.client()


def save_forecast_to_log(db_client, forecast_df):
    """Saves forecast to Firestore for future self-correction learning."""
    if db_client is None or forecast_df.empty:
        return False
    try:
        batch = db_client.batch()
        log_col = db_client.collection('forecast_log')
        gen_time = datetime.now()

        for _, row in forecast_df.iterrows():
            doc_id = row['ds'].strftime('%Y-%m-%d')
            doc_ref = log_col.document(doc_id)
            
            data = {
                'generated_on': gen_time,
                'forecast_for_date': row['ds'],
                'predicted_sales': float(row['forecast_sales']),
                'predicted_customers': int(row['forecast_customers']),
                'predicted_atv': float(row['forecast_atv']),
                'sales_lower': float(row.get('sales_lower', row['forecast_sales'] * 0.9)),
                'sales_upper': float(row.get('sales_upper', row['forecast_sales'] * 1.1)),
                'model_prediction': float(row.get('model_prediction', 0)),
                'sdly_prediction': float(row.get('sdly_prediction', 0)),
                'recent_prediction': float(row.get('recent_prediction', 0)),
                'trend_direction': str(row.get('trend_direction', 'unknown')),
                'trend_confidence': float(row.get('trend_confidence', 0))
            }
            batch.set(doc_ref, data, merge=True)
        
        batch.commit()
        return True
    except Exception as e:
        st.error(f"Firestore Save Error: {e}")
        return False


def create_forecast_chart(forecast_df, hist_df=None):
    """Create interactive forecast visualization."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Sales Forecast', 'Customer Forecast'),
        vertical_spacing=0.12
    )
    
    # Historical data
    if hist_df is not None and len(hist_df) > 0:
        recent_hist = hist_df.tail(30)
        
        fig.add_trace(go.Scatter(
            x=recent_hist['date'], y=recent_hist['sales'],
            mode='lines+markers', name='Actual Sales',
            line=dict(color='#EF4444', width=2), marker=dict(size=6)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=recent_hist['date'], y=recent_hist['customers'],
            mode='lines+markers', name='Actual Customers',
            line=dict(color='#EF4444', width=2), marker=dict(size=6)
        ), row=2, col=1)
    
    # Confidence interval for sales
    if 'sales_lower' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
            y=pd.concat([forecast_df['sales_upper'], forecast_df['sales_lower'][::-1]]),
            fill='toself', fillcolor='rgba(251, 191, 36, 0.2)',
            line=dict(color='rgba(255,255,255,0)'), name='Sales CI', showlegend=True
        ), row=1, col=1)
    
    # Confidence interval for customers
    if 'customers_lower' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
            y=pd.concat([forecast_df['customers_upper'], forecast_df['customers_lower'][::-1]]),
            fill='toself', fillcolor='rgba(251, 191, 36, 0.2)',
            line=dict(color='rgba(255,255,255,0)'), name='Customer CI', showlegend=True
        ), row=2, col=1)
    
    # Forecasts
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'], y=forecast_df['forecast_sales'],
        mode='lines+markers', name='Forecast Sales',
        line=dict(color='#FBBF24', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'], y=forecast_df['forecast_customers'],
        mode='lines+markers', name='Forecast Customers',
        line=dict(color='#FBBF24', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ), row=2, col=1)
    
    # Component predictions
    if 'model_prediction' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=forecast_df['model_prediction'],
            mode='lines', name='Model', line=dict(color='#3B82F6', width=1, dash='dot'), opacity=0.5
        ), row=2, col=1)
    
    if 'recent_prediction' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=forecast_df['recent_prediction'],
            mode='lines', name='Recent Trend', line=dict(color='#8B5CF6', width=1, dash='dot'), opacity=0.5
        ), row=2, col=1)
    
    fig.update_layout(height=600, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
    fig.update_yaxes(title_text="Sales (₱)", row=1, col=1)
    fig.update_yaxes(title_text="Customers", row=2, col=1)
    
    return fig


# === MAIN APP ===
db = init_db()

# Sidebar
with st.sidebar:
    st.title("🧠 Cognitive Forecaster v12.0")
    st.caption("Self-Learning AI with Trend Intelligence + CI")
    
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["📊 Forecast Dashboard", 
         "🎯 Scenarios & Confidence",
         "🧠 Self-Correction Analysis", 
         "📚 Auto-Learning Status",
         "🔬 Backtest & Validate", 
         "📈 Trend Analysis",
         "📋 AI Briefing"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # AI Status
    st.subheader("🤖 AI Status")
    if INTELLIGENT_MODE:
        st.success("✅ Cognitive Mode")
    else:
        st.warning("⚠️ Basic Mode")
    
    if TREND_MODE:
        st.success("✅ Trend Intelligence")
    else:
        st.warning("⚠️ No Trend Module")
    
    st.divider()
    
    # Data Status
    st.subheader("📁 Data Status")
    
    @st.cache_data(ttl=300, show_spinner=False)
    def _cached_load(collection_name):
        """Cache Firestore data for 5 minutes to avoid re-loading on every interaction."""
        return load_from_firestore(db, collection_name)
    
    hist_df = _cached_load('historical_data')
    ev_df = _cached_load('future_activities')
    
    if not hist_df.empty:
        st.success(f"✅ {len(hist_df)} days of data")
        st.caption(f"From: {hist_df['date'].min().strftime('%Y-%m-%d')}")
        st.caption(f"To: {hist_df['date'].max().strftime('%Y-%m-%d')}")
        if len(hist_df) >= 365:
            st.info("✅ SDLY data available")
    else:
        st.error("❌ No historical data")


# ============================================================
# PAGE: Forecast Dashboard
# ============================================================
if page == "📊 Forecast Dashboard":
    st.header("🔮 15-Day Sales Projections")
    st.caption("Trend-Aware Blending with Self-Correction")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
            if len(hist_df) > 364:
                with st.spinner("Training cognitive model..."):
                    progress = st.progress(0, text="Analyzing trends...")
                    
                    try:
                        progress.progress(30, text="Training model...")
                        forecast_df, forecaster, diagnostics = generate_full_forecast(
                            hist_df, ev_df, periods=15, db_client=db
                        )
                        
                        progress.progress(70, text="Saving to database...")
                        save_forecast_to_log(db, forecast_df)
                        
                        progress.progress(100, text="Done!")
                        
                        st.session_state.forecast_df = forecast_df
                        st.session_state.diagnostics = diagnostics
                        st.success("✅ Forecast generated!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.exception(e)
            else:
                st.error(f"Need 365+ days. Current: {len(hist_df)}")
    
    # Display forecast
    if 'forecast_df' in st.session_state:
        forecast_df = st.session_state.forecast_df
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sales", f"₱{forecast_df['forecast_sales'].sum():,.0f}")
        with col2:
            st.metric("Total Customers", f"{forecast_df['forecast_customers'].sum():,}")
        with col3:
            st.metric("Avg ATV", f"₱{forecast_df['forecast_atv'].mean():,.2f}")
        with col4:
            peak = forecast_df.loc[forecast_df['forecast_sales'].idxmax()]
            st.metric("Peak Day", peak['ds'].strftime('%b %d'))
        
        # Trend indicator
        if 'trend_direction' in forecast_df.columns:
            trend = forecast_df['trend_direction'].iloc[0]
            conf = forecast_df.get('trend_confidence', pd.Series([0])).iloc[0]
            
            if trend == 'down':
                st.warning(f"📉 **Downtrend Detected** (Confidence: {conf:.0f}%) - Forecasts adjusted lower")
            elif trend == 'up':
                st.success(f"📈 **Uptrend Detected** (Confidence: {conf:.0f}%) - Positive momentum")
            else:
                st.info("➡️ **Neutral Trend** - Stable forecast")
        
        # Chart
        fig = create_forecast_chart(forecast_df, hist_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("📊 Daily Breakdown")
        display_df = forecast_df.copy()
        display_df['Date'] = display_df['ds'].dt.strftime('%a, %b %d')
        display_df['Sales'] = display_df['forecast_sales'].map("₱{:,.0f}".format)
        display_df['Customers'] = display_df['forecast_customers']
        display_df['ATV'] = display_df['forecast_atv'].map("₱{:,.2f}".format)
        
        if 'trend_direction' in display_df.columns:
            display_df['Trend'] = display_df['trend_direction'].apply(
                lambda x: '📉' if x == 'down' else ('📈' if x == 'up' else '➡️')
            )
            st.dataframe(
                display_df[['Date', 'Sales', 'Customers', 'ATV', 'Trend']].set_index('Date'),
                use_container_width=True
            )
        else:
            st.dataframe(
                display_df[['Date', 'Sales', 'Customers', 'ATV']].set_index('Date'),
                use_container_width=True
            )
        
        # === NEW: Forecast Component Decomposition ===
        if 'model_prediction' in forecast_df.columns and 'sdly_prediction' in forecast_df.columns:
            st.subheader("🔍 Forecast Decomposition")
            st.caption("How each component contributed to the final forecast")
            
            decomp_fig = go.Figure()
            
            decomp_fig.add_trace(go.Scatter(
                x=forecast_df['ds'], y=forecast_df['model_prediction'],
                mode='lines+markers', name='ML Model',
                line=dict(color='#3B82F6', width=2), marker=dict(size=5)
            ))
            
            decomp_fig.add_trace(go.Scatter(
                x=forecast_df['ds'], y=forecast_df['sdly_prediction'],
                mode='lines+markers', name='SDLY (Last Year)',
                line=dict(color='#8B5CF6', width=2), marker=dict(size=5)
            ))
            
            decomp_fig.add_trace(go.Scatter(
                x=forecast_df['ds'], y=forecast_df['recent_prediction'],
                mode='lines+markers', name='Recent Trend',
                line=dict(color='#10B981', width=2), marker=dict(size=5)
            ))
            
            decomp_fig.add_trace(go.Scatter(
                x=forecast_df['ds'], y=forecast_df['forecast_customers'],
                mode='lines+markers', name='Final (Blended)',
                line=dict(color='#FBBF24', width=3), marker=dict(size=8, symbol='diamond')
            ))
            
            decomp_fig.update_layout(
                title='Customer Forecast: Component Breakdown',
                height=400, hovermode='x unified',
                yaxis_title='Customers',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(decomp_fig, use_container_width=True)
            
            # Component agreement metric
            if len(forecast_df) > 0:
                avg_model = forecast_df['model_prediction'].mean()
                avg_sdly = forecast_df['sdly_prediction'].mean()
                avg_recent = forecast_df['recent_prediction'].mean()
                avg_final = forecast_df['forecast_customers'].mean()
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("ML Model Avg", f"{avg_model:,.0f}")
                with col_b:
                    st.metric("SDLY Avg", f"{avg_sdly:,.0f}")
                with col_c:
                    st.metric("Recent Avg", f"{avg_recent:,.0f}")
                with col_d:
                    spread = max(avg_model, avg_sdly, avg_recent) - min(avg_model, avg_sdly, avg_recent)
                    agreement = max(0, 100 - (spread / avg_final * 100))
                    st.metric("Component Agreement", f"{agreement:.0f}%")
    else:
        st.info("👈 Click 'Generate Forecast' to start")


# ============================================================
# PAGE: Scenarios & Confidence
# ============================================================
elif page == "🎯 Scenarios & Confidence":
    st.header("🎯 Multi-Scenario Forecasts")
    st.caption("See optimistic, realistic, and pessimistic scenarios")
    
    if 'forecast_df' in st.session_state:
        forecast_df = st.session_state.forecast_df
        
        # Calculate scenarios based on component predictions
        st.subheader("📊 Scenario Analysis")
        
        scenarios_data = []
        for _, row in forecast_df.iterrows():
            model = row.get('model_prediction', row['forecast_customers'])
            sdly = row.get('sdly_prediction', row['forecast_customers'])
            recent = row.get('recent_prediction', row['forecast_customers'])
            final = row['forecast_customers']
            
            values = [v for v in [model, sdly, recent] if v > 0]
            if values:
                optimistic = int(max(values))
                pessimistic = int(min(values))
                realistic = final
            else:
                optimistic = realistic = pessimistic = final
            
            scenarios_data.append({
                'Date': row['ds'].strftime('%a, %b %d'),
                'Pessimistic': pessimistic,
                'Realistic': realistic,
                'Optimistic': optimistic,
                'Range': f"{pessimistic:,} - {optimistic:,}"
            })
        
        scenarios_df = pd.DataFrame(scenarios_data)
        
        # Scenario chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=[s['Pessimistic'] for s in scenarios_data],
            mode='lines', name='Pessimistic', line=dict(color='#EF4444', dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=[s['Realistic'] for s in scenarios_data],
            mode='lines+markers', name='Realistic', line=dict(color='#3B82F6', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=[s['Optimistic'] for s in scenarios_data],
            mode='lines', name='Optimistic', line=dict(color='#10B981', dash='dot'),
            fill='tonexty', fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        
        fig.update_layout(title='Customer Forecast Scenarios', height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence scoring
        st.subheader("🎯 Confidence Scores")
        
        confidence_data = []
        for _, row in forecast_df.iterrows():
            model = row.get('model_prediction', 0)
            sdly = row.get('sdly_prediction', 0)
            recent = row.get('recent_prediction', 0)
            final = row['forecast_customers']
            
            # Calculate agreement between components
            values = [v for v in [model, sdly, recent] if v > 0]
            if len(values) >= 2:
                max_diff = max(values) - min(values)
                mean_val = np.mean(values)
                agreement = 1 - (max_diff / mean_val) if mean_val > 0 else 0
                confidence = min(100, int(agreement * 100 + 20))
            else:
                confidence = 50
            
            grade = 'A' if confidence >= 80 else 'B' if confidence >= 65 else 'C' if confidence >= 50 else 'D'
            
            confidence_data.append({
                'Date': row['ds'].strftime('%a, %b %d'),
                'Forecast': f"{final:,}",
                'Confidence': f"{confidence}%",
                'Grade': grade
            })
        
        conf_df = pd.DataFrame(confidence_data)
        st.dataframe(conf_df.set_index('Date'), use_container_width=True)
        
        # Summary
        avg_conf = np.mean([int(c['Confidence'].replace('%', '')) for c in confidence_data])
        st.metric("Average Confidence", f"{avg_conf:.0f}%")
        
    else:
        st.info("Generate a forecast first to see scenarios")


# ============================================================
# PAGE: Self-Correction Analysis
# ============================================================
elif page == "🧠 Self-Correction Analysis":
    st.header("🧠 Self-Correction Learning")
    st.caption("See what the AI has learned from past mistakes")
    
    if 'diagnostics' in st.session_state:
        diag = st.session_state.diagnostics
        blend_diag = diag.get('blend_diagnostics', {})
        self_corr = blend_diag.get('self_correction', {})
        
        if self_corr.get('is_calibrated', False):
            st.success("✅ Self-correction is ACTIVE")
            
            correction_factors = self_corr.get('correction_factors', {})
            error_patterns = self_corr.get('error_patterns', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", error_patterns.get('n_samples', 0))
            with col2:
                st.metric("Historical MAPE", f"{error_patterns.get('overall_mape', 0):.1f}%")
            with col3:
                overall = correction_factors.get('overall', 1.0)
                bias = "Under" if overall > 1 else "Over"
                st.metric("Detected Bias", f"{bias}-predicting")
            
            # Day-of-week corrections
            st.subheader("📅 Day-of-Week Corrections")
            dow_corr = correction_factors.get('dow', {})
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            if dow_corr:
                dow_data = []
                for dow, corr in sorted(dow_corr.items()):
                    dow_data.append({
                        'Day': dow_names[int(dow)],
                        'Factor': f"{corr:.3f}",
                        'Adjustment': f"{(corr-1)*100:+.1f}%",
                        'Direction': '📈 Increase' if corr > 1.01 else ('📉 Decrease' if corr < 0.99 else '➡️ None')
                    })
                st.dataframe(pd.DataFrame(dow_data).set_index('Day'), use_container_width=True)
            
            # Payday correction
            st.subheader("💰 Special Corrections")
            payday = correction_factors.get('payday', 1.0)
            st.write(f"**Payday Adjustment:** {(payday-1)*100:+.1f}%")
            
        else:
            st.warning("⚠️ Self-correction not yet calibrated")
            st.info("""
            **How to enable:**
            1. Generate forecasts regularly
            2. Wait for actual data to come in
            3. System compares and learns automatically
            4. After 7+ comparisons, self-correction activates
            """)
    else:
        st.info("Generate a forecast first to see self-correction data")


# ============================================================
# PAGE: Auto-Learning Status
# ============================================================
elif page == "📚 Auto-Learning Status":
    st.header("📚 Automatic Learning System")
    st.caption("Monitor AI's continuous learning")
    
    # Explanation
    with st.expander("ℹ️ How Auto-Learning Works"):
        st.markdown("""
        **The learning cycle compares past forecasts to actual results:**
        
        1. When you generate a forecast, it's saved to `forecast_log`
        2. When actual data comes in (days pass), we can compare
        3. The system learns from the errors and adjusts future predictions
        
        **Requirements:**
        - Forecasts must be for dates that NOW have actual data
        - Example: A forecast made on Jan 15 for Jan 20 can be evaluated after Jan 20
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🔧 Actions")
        
        if st.button("🔄 Run Learning Cycle", type="primary", use_container_width=True):
            with st.spinner("Running learning cycle..."):
                try:
                    # Load forecast log
                    forecast_docs = list(db.collection('forecast_log').stream())
                    
                    st.write(f"📊 Found {len(forecast_docs)} forecast records")
                    
                    if len(forecast_docs) == 0:
                        st.warning("No forecasts in database yet. Generate some forecasts first!")
                    else:
                        forecasts = []
                        parse_errors = []
                        
                        for doc in forecast_docs:
                            data = doc.to_dict()
                            doc_id = doc.id
                            
                            # Try to get the forecast target date
                            fc_date = None
                            
                            # Option 1: Use 'forecast_for_date' field
                            if fc_date is None and 'forecast_for_date' in data:
                                try:
                                    val = data['forecast_for_date']
                                    if hasattr(val, 'date'):  # Firestore timestamp
                                        fc_date = pd.to_datetime(val.date())
                                    else:
                                        fc_date = pd.to_datetime(val)
                                except Exception as e:
                                    parse_errors.append(f"forecast_for_date: {e}")
                            
                            # Option 2: Parse document ID
                            if fc_date is None:
                                try:
                                    if '_' in doc_id:
                                        date_part = doc_id.split('_')[0]
                                        fc_date = pd.to_datetime(date_part)
                                    else:
                                        fc_date = pd.to_datetime(doc_id)
                                except Exception as e:
                                    parse_errors.append(f"doc_id {doc_id}: {e}")
                            
                            if fc_date is not None:
                                # Normalize to remove time component
                                fc_date = fc_date.normalize() if hasattr(fc_date, 'normalize') else pd.to_datetime(fc_date.date())
                                
                                forecasts.append({
                                    'forecast_date': fc_date,
                                    'predicted_customers': data.get('predicted_customers', 0),
                                    'predicted_sales': data.get('predicted_sales', 0),
                                    'doc_id': doc_id
                                })
                        
                        if parse_errors:
                            with st.expander(f"⚠️ {len(parse_errors)} parse warnings"):
                                for err in parse_errors[:10]:
                                    st.write(err)
                        
                        st.write(f"✅ Successfully parsed {len(forecasts)} forecasts")
                        
                        # Get date range of forecasts
                        if forecasts:
                            forecast_df = pd.DataFrame(forecasts)
                            min_fc_date = forecast_df['forecast_date'].min()
                            max_fc_date = forecast_df['forecast_date'].max()
                            st.write(f"📅 Forecast dates: {min_fc_date.strftime('%Y-%m-%d')} to {max_fc_date.strftime('%Y-%m-%d')}")
                            
                            # Get date range of actual data
                            hist_min = hist_df['date'].min()
                            hist_max = hist_df['date'].max()
                            st.write(f"📅 Actual data: {hist_min.strftime('%Y-%m-%d')} to {hist_max.strftime('%Y-%m-%d')}")
                            
                            # Normalize historical dates
                            hist_df['date_normalized'] = hist_df['date'].dt.normalize()
                            
                            # Find matches
                            matches = 0
                            total_error = 0
                            errors_list = []
                            
                            for _, fc in forecast_df.iterrows():
                                fc_date = fc['forecast_date']
                                
                                # Check if this forecast date has actual data
                                actual = hist_df[hist_df['date_normalized'] == fc_date]
                                
                                if len(actual) > 0:
                                    pred = fc['predicted_customers']
                                    act = actual.iloc[0]['customers']
                                    
                                    if act > 0 and pred > 0:
                                        error = abs(act - pred) / act * 100
                                        signed_error = (act - pred) / act * 100
                                        total_error += error
                                        matches += 1
                                        
                                        errors_list.append({
                                            'date': fc_date,
                                            'predicted': pred,
                                            'actual': act,
                                            'error_pct': error,
                                            'signed_error': signed_error
                                        })
                            
                            st.write(f"🔗 Matched {matches} forecasts with actual data")
                            
                            if matches > 0:
                                avg_mape = total_error / matches
                                accuracy = 100 - avg_mape
                                avg_bias = np.mean([e['signed_error'] for e in errors_list])
                                
                                st.session_state.learning_result = {
                                    'matches': matches,
                                    'mape': avg_mape,
                                    'accuracy': accuracy,
                                    'bias': avg_bias,
                                    'errors': errors_list
                                }
                                st.success(f"✅ Learning cycle complete!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                # Show why no matches
                                st.warning("No matching actual data found.")
                                
                                # Check if forecasts are for future dates
                                future_forecasts = forecast_df[forecast_df['forecast_date'] > hist_max]
                                if len(future_forecasts) > 0:
                                    st.info(f"""
                                    📌 **Why no matches?**
                                    
                                    {len(future_forecasts)} of your forecasts are for **future dates** 
                                    (after {hist_max.strftime('%Y-%m-%d')}).
                                    
                                    The learning system can only evaluate forecasts once 
                                    actual data is available.
                                    
                                    **Solution:** Wait for actual data to come in, or use 
                                    the Backtest page to test on historical data.
                                    """)
                                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.divider()
        
        # Alternative: Learn from backtest
        if st.button("📊 Learn from Backtest", use_container_width=True):
            with st.spinner("Running backtest learning..."):
                try:
                    # Run a backtest and use those results for learning
                    results = backtest_model(hist_df, ev_df, test_days=30, step_size=7)
                    
                    if len(results) > 0:
                        errors_list = []
                        for _, row in results.iterrows():
                            pred = row['predicted_customers']
                            act = row['actual_customers']
                            if act > 0 and pred > 0:
                                error = abs(act - pred) / act * 100
                                signed_error = (act - pred) / act * 100
                                errors_list.append({
                                    'date': row['date'],
                                    'predicted': pred,
                                    'actual': act,
                                    'error_pct': error,
                                    'signed_error': signed_error
                                })
                        
                        if errors_list:
                            avg_mape = np.mean([e['error_pct'] for e in errors_list])
                            accuracy = 100 - avg_mape
                            avg_bias = np.mean([e['signed_error'] for e in errors_list])
                            
                            st.session_state.learning_result = {
                                'matches': len(errors_list),
                                'mape': avg_mape,
                                'accuracy': accuracy,
                                'bias': avg_bias,
                                'errors': errors_list,
                                'source': 'backtest'
                            }
                            st.success("✅ Learning from backtest complete!")
                            time.sleep(1)
                            st.rerun()
                except Exception as e:
                    st.error(f"Backtest error: {e}")
    
    with col1:
        # Display results
        if 'learning_result' in st.session_state:
            result = st.session_state.learning_result
            
            source = result.get('source', 'forecast_log')
            st.subheader(f"📊 Learning Results (from {source})")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Predictions Analyzed", result['matches'])
            with col_b:
                st.metric("Average MAPE", f"{result['mape']:.1f}%")
            with col_c:
                acc = result['accuracy']
                st.metric("Accuracy", f"{acc:.1f}%")
            with col_d:
                bias = result.get('bias', 0)
                bias_dir = "Under" if bias > 0 else "Over"
                st.metric("Bias Direction", f"{bias_dir} {abs(bias):.1f}%")
            
            # Accuracy assessment
            if result['accuracy'] >= 92:
                st.success("🎉 **Excellent accuracy!** Model is performing very well.")
            elif result['accuracy'] >= 88:
                st.info("👍 **Good accuracy.** Minor improvements possible.")
            elif result['accuracy'] >= 85:
                st.warning("⚠️ **Moderate accuracy.** Consider reviewing model parameters.")
            else:
                st.error("🔴 **Accuracy needs improvement.** Recommend investigation.")
            
            # Bias interpretation
            bias = result.get('bias', 0)
            if abs(bias) > 5:
                if bias > 0:
                    st.warning(f"📊 **Systematic under-prediction detected:** The model predicts {abs(bias):.1f}% lower than actual on average. Self-correction will increase future predictions.")
                else:
                    st.warning(f"📊 **Systematic over-prediction detected:** The model predicts {abs(bias):.1f}% higher than actual on average. Self-correction will decrease future predictions.")
            
            # Show error details
            if 'errors' in result and len(result['errors']) > 0:
                st.subheader("📋 Prediction Details")
                
                errors_df = pd.DataFrame(result['errors'])
                errors_df['Date'] = pd.to_datetime(errors_df['date']).dt.strftime('%Y-%m-%d')
                errors_df['Predicted'] = errors_df['predicted'].astype(int)
                errors_df['Actual'] = errors_df['actual'].astype(int)
                errors_df['Error'] = errors_df['error_pct'].apply(lambda x: f"{x:.1f}%")
                errors_df['Direction'] = errors_df['signed_error'].apply(
                    lambda x: '📉 Under' if x > 2 else ('📈 Over' if x < -2 else '✅ Good')
                )
                
                st.dataframe(
                    errors_df[['Date', 'Predicted', 'Actual', 'Error', 'Direction']].set_index('Date'),
                    use_container_width=True
                )
                
                # Error distribution chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=errors_df['Date'],
                    y=errors_df['signed_error'],
                    marker_color=errors_df['signed_error'].apply(
                        lambda x: '#EF4444' if x > 5 else ('#10B981' if x < -5 else '#3B82F6')
                    ),
                    name='Error %'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title='Prediction Errors (Positive = Under-predicted)',
                    xaxis_title='Date',
                    yaxis_title='Error %',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Click 'Run Learning Cycle' or 'Learn from Backtest' to analyze model performance.")
    
    # Show learning history from Firestore
    st.divider()
    st.subheader("📜 Stored Learning Data")
    
    try:
        learning_docs = list(db.collection('ai_learning').stream())
        
        if learning_docs:
            learning_data = {doc.id: doc.to_dict() for doc in learning_docs}
            for doc_name, data in learning_data.items():
                with st.expander(f"📄 {doc_name}"):
                    st.json(data)
        else:
            st.info("No learning data stored yet. Complete a learning cycle to save corrections.")
    except Exception as e:
        st.info("No learning data available yet.")


# ============================================================
# PAGE: Backtest & Validate
# ============================================================
elif page == "🔬 Backtest & Validate":
    st.header("🔬 Model Validation")
    st.caption("Test accuracy on historical data")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        test_days = st.slider("Test Days", 14, 60, 30)
        
        if st.button("🧪 Run Backtest", type="primary", use_container_width=True):
            if len(hist_df) > 400:
                with st.spinner(f"Backtesting {test_days} days..."):
                    try:
                        results = backtest_model(hist_df, ev_df, test_days=test_days, step_size=7)
                        
                        if len(results) > 0:
                            st.session_state.backtest_results = results
                            metrics = calculate_accuracy_metrics(results)
                            st.session_state.backtest_metrics = metrics
                            st.success(f"✅ Tested {len(results)} predictions")
                        else:
                            st.warning("No results generated")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Need 400+ days of data")
    
    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        metrics = st.session_state.backtest_metrics
        
        overall = metrics.get('overall', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Customer Accuracy", f"{overall.get('customer_accuracy', 0):.1f}%")
        with col2:
            st.metric("Sales Accuracy", f"{overall.get('sales_accuracy', 0):.1f}%")
        with col3:
            st.metric("Bias", f"{overall.get('customer_bias', 0):+.1f}%")
        with col4:
            st.metric("Samples", overall.get('n_predictions', 0))
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['date'], y=results['actual_customers'],
            mode='lines+markers', name='Actual', line=dict(color='#EF4444')
        ))
        fig.add_trace(go.Scatter(
            x=results['date'], y=results['predicted_customers'],
            mode='lines+markers', name='Predicted', line=dict(color='#FBBF24', dash='dash')
        ))
        fig.update_layout(title='Backtest: Actual vs Predicted', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click 'Run Backtest' to test model accuracy")


# ============================================================
# PAGE: Trend Analysis
# ============================================================
elif page == "📈 Trend Analysis":
    st.header("📈 Trend Intelligence Analysis")
    st.caption("Deep analysis of trend signals")
    
    if not hist_df.empty and TREND_MODE:
        target_date = hist_df['date'].max() + timedelta(days=1)
        
        st.subheader("🔍 Trend Signal Analysis")
        st.write(f"**Analyzing trends for:** {target_date.strftime('%A, %B %d, %Y')}")
        
        # Run trend analysis
        try:
            analyzer = TrendAnalyzer()
            analysis = analyzer.analyze_comprehensive_trend(hist_df, target_date)
            
            # Overall trend
            direction = analysis['overall_direction']
            confidence = analysis['confidence']
            
            if direction == 'down':
                st.error(f"📉 **Overall Trend: DOWN** (Confidence: {confidence:.0f}%)")
            elif direction == 'up':
                st.success(f"📈 **Overall Trend: UP** (Confidence: {confidence:.0f}%)")
            else:
                st.info(f"➡️ **Overall Trend: NEUTRAL** (Confidence: {confidence:.0f}%)")
            
            # Individual signals
            st.subheader("📊 Individual Signals")
            
            signals = analysis['signals']
            
            for signal_name, signal_data in signals.items():
                if signal_data.get('data_available', False):
                    direction = signal_data.get('direction', 'neutral')
                    strength = signal_data.get('strength', 0)
                    
                    css_class = f"signal-{direction}"
                    icon = '📉' if direction == 'down' else ('📈' if direction == 'up' else '➡️')
                    
                    st.markdown(f"""
                    <div class="{css_class}">
                        <strong>{icon} {signal_name.replace('_', ' ').title()}</strong><br>
                        Direction: {direction.upper()} | Strength: {strength:.2f}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommended adjustment
            st.subheader("🎯 Recommended Adjustment")
            adj = analysis.get('recommended_adjustment', 1.0)
            if adj < 1:
                st.warning(f"Reduce forecasts by **{(1-adj)*100:.1f}%**")
            elif adj > 1:
                st.success(f"Increase forecasts by **{(adj-1)*100:.1f}%**")
            else:
                st.info("No adjustment needed")
                
        except Exception as e:
            st.error(f"Trend analysis error: {e}")
    
    elif not TREND_MODE:
        st.warning("Trend Intelligence module not available")
    else:
        st.info("No historical data available")
    
    # Recent performance chart
    st.subheader("📊 Recent 8-Week Performance")
    
    if not hist_df.empty:
        recent_8w = hist_df.tail(56)
        
        fig = px.line(recent_8w, x='date', y='customers', 
                     title='Customer Traffic - Last 8 Weeks')
        fig.add_hline(y=recent_8w['customers'].mean(), line_dash="dash", 
                     annotation_text=f"Avg: {recent_8w['customers'].mean():.0f}")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: AI Briefing
# ============================================================
elif page == "📋 AI Briefing":
    st.header("📋 AI Executive Briefing")
    st.caption("Automated daily briefing")
    
    if 'forecast_df' in st.session_state:
        forecast_df = st.session_state.forecast_df
        
        # Generate briefing
        total_sales = forecast_df['forecast_sales'].sum()
        total_customers = forecast_df['forecast_customers'].sum()
        avg_daily_sales = total_sales / len(forecast_df)
        peak_day = forecast_df.loc[forecast_df['forecast_sales'].idxmax()]
        
        st.markdown(f"""
        <div class="briefing-box">
        <h2>📊 15-Day Forecast Summary</h2>
        <table style="width:100%; color:white;">
            <tr><td><strong>Total Projected Sales</strong></td><td style="text-align:right">₱{total_sales:,.0f}</td></tr>
            <tr><td><strong>Total Projected Customers</strong></td><td style="text-align:right">{total_customers:,}</td></tr>
            <tr><td><strong>Average Daily Sales</strong></td><td style="text-align:right">₱{avg_daily_sales:,.0f}</td></tr>
            <tr><td><strong>Peak Day</strong></td><td style="text-align:right">{peak_day['ds'].strftime('%A, %B %d')}</td></tr>
            <tr><td><strong>Peak Day Sales</strong></td><td style="text-align:right">₱{peak_day['forecast_sales']:,.0f}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Trend insight
        st.subheader("📈 Trend Insight")
        
        first_week = forecast_df.head(7)['forecast_sales'].mean()
        second_week = forecast_df.tail(8)['forecast_sales'].mean()
        week_change = (second_week - first_week) / first_week * 100
        
        if week_change > 5:
            st.success(f"📈 **Growing Trend:** Week 2 projected {week_change:.1f}% higher than Week 1")
        elif week_change < -5:
            st.warning(f"📉 **Declining Trend:** Week 2 projected {abs(week_change):.1f}% lower than Week 1")
        else:
            st.info(f"➡️ **Stable Trend:** Relatively flat ({week_change:+.1f}%)")
        
        # Key alerts
        st.subheader("🔔 Key Alerts")
        
        if 'trend_direction' in forecast_df.columns:
            downs = (forecast_df['trend_direction'] == 'down').sum()
            if downs > 10:
                st.warning(f"⚠️ {downs} of 15 days show downtrend signals")
        
        # Low days
        avg_cust = forecast_df['forecast_customers'].mean()
        low_days = forecast_df[forecast_df['forecast_customers'] < avg_cust * 0.9]
        if len(low_days) > 0:
            st.info(f"📉 {len(low_days)} days below average expected")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        recommendations = []
        
        if week_change < -5:
            recommendations.append("Consider promotional activities to counter declining trend")
        
        if 'trend_direction' in forecast_df.columns and forecast_df['trend_direction'].iloc[0] == 'down':
            recommendations.append("Conservative staffing recommended due to detected weakness")
        
        if peak_day['ds'].weekday() >= 5:
            recommendations.append(f"Ensure adequate weekend staffing for {peak_day['ds'].strftime('%A')}")
        
        if not recommendations:
            recommendations.append("No special actions needed - forecast looks stable")
        
        for rec in recommendations:
            st.write(f"• {rec}")
            
    else:
        st.info("Generate a forecast first to see the AI briefing")


# Footer
st.divider()
st.caption("Cognitive Forecasting Engine v12.0 | Trend-Aware Self-Learning AI with Confidence Intervals")
