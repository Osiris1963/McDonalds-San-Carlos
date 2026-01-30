# app.py - ENHANCED VERSION v10.0
# Key enhancements:
# 1. Self-Correction Dashboard - shows learned biases
# 2. Prediction Component Analysis - shows SDLY vs Recent vs Model
# 3. Contextual Window Visualization
# 4. 8-Week Trend Analysis Display

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import timedelta

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
    EnsembleForecaster
)

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Sales Forecaster v10.0", 
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
    .component-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 10px 0;
    }
    .self-correct-positive { color: #10B981; }
    .self-correct-negative { color: #EF4444; }
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
        gen_time = pd.to_datetime('now')

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
                # Store component predictions for analysis
                'model_prediction': float(row.get('model_prediction', 0)),
                'sdly_prediction': float(row.get('sdly_prediction', 0)),
                'recent_prediction': float(row.get('recent_prediction', 0)),
                'yoy_growth': float(row.get('yoy_growth', 1.0))
            }
            batch.set(doc_ref, data, merge=True)
        
        batch.commit()
        return True
    except Exception as e:
        st.error(f"Firestore Save Error: {e}")
        return False


def create_forecast_chart(forecast_df, hist_df=None, show_confidence=True):
    """Create interactive forecast visualization with component breakdown."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Sales Forecast', 'Customer Forecast - Component Breakdown'),
        vertical_spacing=0.12
    )
    
    # Historical data
    if hist_df is not None and len(hist_df) > 0:
        recent_hist = hist_df.tail(30)
        
        fig.add_trace(
            go.Scatter(
                x=recent_hist['date'], 
                y=recent_hist['sales'],
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='#EF4444', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=recent_hist['date'], 
                y=recent_hist['customers'],
                mode='lines+markers',
                name='Actual Customers',
                line=dict(color='#EF4444', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
    
    # Confidence interval
    if show_confidence and 'sales_lower' in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
                y=pd.concat([forecast_df['sales_upper'], forecast_df['sales_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(251, 191, 36, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Sales 80% CI',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Sales forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df['ds'], 
            y=forecast_df['forecast_sales'],
            mode='lines+markers',
            name='Forecast Sales',
            line=dict(color='#FBBF24', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ),
        row=1, col=1
    )
    
    # Customer forecast with components
    fig.add_trace(
        go.Scatter(
            x=forecast_df['ds'], 
            y=forecast_df['forecast_customers'],
            mode='lines+markers',
            name='Final Forecast',
            line=dict(color='#FBBF24', width=3),
            marker=dict(size=10, symbol='diamond')
        ),
        row=2, col=1
    )
    
    # Show component predictions if available
    if 'model_prediction' in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['model_prediction'],
                mode='lines',
                name='Model Component',
                line=dict(color='#3B82F6', width=1, dash='dot'),
                opacity=0.6
            ),
            row=2, col=1
        )
    
    if 'sdly_prediction' in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['sdly_prediction'],
                mode='lines',
                name='SDLY Component',
                line=dict(color='#10B981', width=1, dash='dot'),
                opacity=0.6
            ),
            row=2, col=1
        )
    
    if 'recent_prediction' in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['recent_prediction'],
                mode='lines',
                name='Recent Trend Component',
                line=dict(color='#8B5CF6', width=1, dash='dot'),
                opacity=0.6
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=700,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sales (₱)", row=1, col=1)
    fig.update_yaxes(title_text="Customers", row=2, col=1)
    
    return fig


def create_contextual_window_chart(hist_df, target_date):
    """Visualize the SDLY ± 14 day contextual window."""
    sdly_date = target_date - timedelta(days=364)
    window_start = sdly_date - timedelta(days=14)
    window_end = sdly_date + timedelta(days=14)
    
    window_data = hist_df[(hist_df['date'] >= window_start) & (hist_df['date'] <= window_end)]
    
    if window_data.empty:
        return None
    
    fig = go.Figure()
    
    # Window data
    fig.add_trace(go.Scatter(
        x=window_data['date'],
        y=window_data['customers'],
        mode='lines+markers',
        name='Last Year Data',
        line=dict(color='#3B82F6', width=2)
    ))
    
    # Highlight SDLY date
    sdly_data = window_data[window_data['date'] == sdly_date]
    if len(sdly_data) > 0:
        fig.add_trace(go.Scatter(
            x=[sdly_date],
            y=[sdly_data['customers'].values[0]],
            mode='markers',
            name='SDLY (Same Day Last Year)',
            marker=dict(color='#EF4444', size=15, symbol='star')
        ))
    
    # Add vertical line for SDLY
    fig.add_vline(x=sdly_date, line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title=f"Contextual Window: {window_start.strftime('%b %d')} - {window_end.strftime('%b %d, %Y')}",
        xaxis_title="Date (Last Year)",
        yaxis_title="Customers",
        height=350
    )
    
    return fig


def create_recent_trend_chart(hist_df, weeks=8):
    """Visualize the 8-week recent trend with week-over-week comparison."""
    end_date = hist_df['date'].max()
    start_date = end_date - timedelta(weeks=weeks)
    
    recent_data = hist_df[(hist_df['date'] > start_date) & (hist_df['date'] <= end_date)].copy()
    
    if recent_data.empty:
        return None
    
    recent_data['weeks_ago'] = ((end_date - recent_data['date']).dt.days // 7) + 1
    weekly_agg = recent_data.groupby('weeks_ago').agg({
        'customers': 'mean',
        'sales': 'mean'
    }).reset_index().sort_values('weeks_ago', ascending=False)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Weekly Customer Average', 'Trend Direction'))
    
    # Bar chart of weekly averages
    colors = ['#10B981' if i <= 2 else '#3B82F6' if i <= 4 else '#9CA3AF' 
              for i in weekly_agg['weeks_ago']]
    
    fig.add_trace(
        go.Bar(
            x=[f"Week {int(w)}" for w in weekly_agg['weeks_ago']],
            y=weekly_agg['customers'],
            marker_color=colors,
            name='Avg Customers'
        ),
        row=1, col=1
    )
    
    # Trend line
    fig.add_trace(
        go.Scatter(
            x=[f"Week {int(w)}" for w in weekly_agg['weeks_ago']],
            y=weekly_agg['customers'],
            mode='lines+markers',
            name='Trend',
            line=dict(color='#EF4444', width=2)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        title_text="8-Week Recent Performance (Week 1 = Most Recent)"
    )
    
    return fig


def create_backtest_chart(backtest_df, metric='customers'):
    """Create chart comparing predictions vs actuals from backtesting."""
    fig = go.Figure()
    
    actual_col = f'actual_{metric}'
    pred_col = f'predicted_{metric}'
    
    fig.add_trace(go.Scatter(
        x=backtest_df['date'],
        y=backtest_df[actual_col],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#EF4444', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=backtest_df['date'],
        y=backtest_df[pred_col],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='#FBBF24', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{metric.title()} - Backtest Results',
        xaxis_title='Date',
        yaxis_title=metric.title(),
        hovermode='x unified'
    )
    
    return fig


# === MAIN APP ===
db = init_db()

# Sidebar
with st.sidebar:
    st.title("🚀 Smart Forecaster v10.0")
    st.caption("Self-Correcting AI with Contextual Analysis")
    
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["📊 Forecast Dashboard", "🧠 Self-Correction Analysis", "🔬 Backtest & Validate", "📈 Trend Analysis"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Data info
    st.subheader("📁 Data Status")
    hist_df = load_from_firestore(db, 'historical_data')
    ev_df = load_from_firestore(db, 'future_activities')
    
    if not hist_df.empty:
        st.success(f"✅ {len(hist_df)} days of data")
        st.caption(f"From: {hist_df['date'].min().strftime('%Y-%m-%d')}")
        st.caption(f"To: {hist_df['date'].max().strftime('%Y-%m-%d')}")
        
        if len(hist_df) >= 365:
            st.info("✅ SDLY data available")
        else:
            st.warning(f"⚠️ Need {365 - len(hist_df)} more days for SDLY")
    else:
        st.error("❌ No historical data found")


# === PAGE: Forecast Dashboard ===
if page == "📊 Forecast Dashboard":
    st.header("🔮 15-Day Sales Projections")
    st.caption("Blended Prediction: 50% Model + 25% SDLY Context + 25% Recent Trend")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
            if len(hist_df) > 364:
                with st.spinner("Training self-correcting ensemble model..."):
                    progress = st.progress(0, text="Preparing data...")
                    
                    try:
                        progress.progress(20, text="Calibrating self-correction...")
                        forecast_df, forecaster, diagnostics = generate_full_forecast(
                            hist_df, ev_df, periods=15, db_client=db
                        )
                        
                        progress.progress(70, text="Saving to Firestore...")
                        success = save_forecast_to_log(db, forecast_df)
                        
                        progress.progress(100, text="Done!")
                        
                        if success:
                            st.session_state.forecast_df = forecast_df
                            st.session_state.diagnostics = diagnostics
                            st.success("✅ Forecast generated and saved!")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Forecast Error: {e}")
                        st.exception(e)
            else:
                st.error(f"Need at least 365 days of data. Current: {len(hist_df)} days.")
    
    # Display forecast
    if 'forecast_df' in st.session_state:
        forecast_df = st.session_state.forecast_df
        
        # Summary metrics
        st.subheader("📋 Forecast Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_sales = forecast_df['forecast_sales'].sum()
        total_customers = forecast_df['forecast_customers'].sum()
        avg_atv = forecast_df['forecast_atv'].mean()
        peak_day = forecast_df.loc[forecast_df['forecast_sales'].idxmax()]
        
        with col1:
            st.metric("Total Projected Sales", f"₱{total_sales:,.0f}")
        with col2:
            st.metric("Total Projected Customers", f"{total_customers:,}")
        with col3:
            st.metric("Average ATV", f"₱{avg_atv:,.2f}")
        with col4:
            st.metric("Peak Day", peak_day['ds'].strftime('%b %d'))
        
        # Chart
        st.subheader("📈 Forecast Visualization")
        fig = create_forecast_chart(forecast_df, hist_df, show_confidence=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Component breakdown
        st.subheader("🔍 Prediction Component Breakdown")
        
        display_df = forecast_df.copy()
        display_df['Date'] = display_df['ds'].dt.strftime('%a, %b %d')
        display_df['Final Forecast'] = display_df['forecast_customers']
        display_df['Model (50%)'] = display_df['model_prediction'].round(0).astype(int)
        display_df['SDLY (25%)'] = display_df['sdly_prediction'].round(0).astype(int)
        display_df['Recent (25%)'] = display_df['recent_prediction'].round(0).astype(int)
        display_df['YoY Growth'] = display_df['yoy_growth'].map("{:.1%}".format)
        display_df['Sales'] = display_df['forecast_sales'].map("₱{:,.0f}".format)
        
        st.dataframe(
            display_df[['Date', 'Final Forecast', 'Model (50%)', 'SDLY (25%)', 'Recent (25%)', 'YoY Growth', 'Sales']].set_index('Date'),
            use_container_width=True
        )
    else:
        st.info("👈 Click 'Generate Forecast' to create your 15-day outlook")


# === PAGE: Self-Correction Analysis ===
elif page == "🧠 Self-Correction Analysis":
    st.header("🧠 Self-Correction Learning")
    st.caption("The AI learns from past forecast errors and adjusts future predictions")
    
    if 'diagnostics' in st.session_state:
        diag = st.session_state.diagnostics
        blend_diag = diag.get('blend_diagnostics', {})
        self_corr = blend_diag.get('self_correction', {})
        
        if self_corr.get('is_calibrated', False):
            st.success("✅ Self-correction is ACTIVE and calibrated")
            
            correction_factors = self_corr.get('correction_factors', {})
            error_patterns = self_corr.get('error_patterns', {})
            
            # Overall stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_samples = error_patterns.get('n_samples', 0)
                st.metric("Training Samples", n_samples)
            
            with col2:
                overall_mape = error_patterns.get('overall_mape', 0)
                st.metric("Historical MAPE", f"{overall_mape:.1f}%")
            
            with col3:
                overall_corr = correction_factors.get('overall', 1.0)
                bias_dir = "Under-predicting" if overall_corr > 1 else "Over-predicting"
                st.metric("Detected Bias", bias_dir)
            
            # Day-of-week corrections
            st.subheader("📅 Day-of-Week Correction Factors")
            
            dow_corr = correction_factors.get('dow', {})
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            if dow_corr:
                dow_df = pd.DataFrame([
                    {
                        'Day': dow_names[dow],
                        'Correction Factor': f"{corr:.3f}",
                        'Adjustment': f"{(corr - 1) * 100:+.1f}%",
                        'Direction': '📈 Increase' if corr > 1.01 else ('📉 Decrease' if corr < 0.99 else '➡️ No change')
                    }
                    for dow, corr in sorted(dow_corr.items())
                ])
                st.dataframe(dow_df.set_index('Day'), use_container_width=True)
            
            # Month corrections
            st.subheader("📆 Monthly Correction Factors")
            
            month_corr = correction_factors.get('month', {})
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            if month_corr:
                month_data = []
                for month, corr in sorted(month_corr.items()):
                    month_data.append({
                        'Month': month_names[month - 1],
                        'Correction': corr,
                        'Adjustment': f"{(corr - 1) * 100:+.1f}%"
                    })
                
                fig = px.bar(
                    month_data, x='Month', y='Correction',
                    title='Monthly Correction Factors (1.0 = no adjustment)',
                    color='Correction',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=1.0
                )
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            
            # Payday correction
            st.subheader("💰 Special Date Corrections")
            
            payday_corr = correction_factors.get('payday', 1.0)
            st.markdown(f"""
            - **Payday Correction**: {payday_corr:.3f} ({(payday_corr - 1) * 100:+.1f}%)
            - **Overall Bias Correction**: {correction_factors.get('overall', 1.0):.3f}
            """)
            
        else:
            st.warning("⚠️ Self-correction not yet calibrated")
            st.info("""
            Self-correction requires historical forecast vs actual comparisons.
            
            **How to enable:**
            1. Generate forecasts regularly
            2. Wait for actual data to come in
            3. The system automatically compares and learns
            4. After 7+ comparisons, self-correction activates
            """)
    else:
        st.info("👈 Generate a forecast first to see self-correction analysis")


# === PAGE: Backtest & Validate ===
elif page == "🔬 Backtest & Validate":
    st.header("🔬 Model Validation & Backtesting")
    st.caption("Test model accuracy on historical data before deploying")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        test_days = st.slider("Test Period (days)", 14, 90, 30)
        
        if st.button("🧪 Run Backtest", type="primary", use_container_width=True):
            if len(hist_df) > 400:
                with st.spinner(f"Running walk-forward backtest on last {test_days} days..."):
                    try:
                        backtest_results = backtest_model(
                            hist_df, ev_df, 
                            db_client=None,  # No self-correction during backtest
                            test_days=test_days, 
                            step_size=7
                        )
                        
                        if len(backtest_results) > 0:
                            st.session_state.backtest_results = backtest_results
                            st.session_state.accuracy_metrics = calculate_accuracy_metrics(backtest_results)
                            st.success(f"✅ Backtest complete! {len(backtest_results)} predictions evaluated.")
                        else:
                            st.warning("No backtest results generated.")
                    except Exception as e:
                        st.error(f"Backtest Error: {e}")
                        st.exception(e)
            else:
                st.error(f"Need at least 400 days for backtesting. Current: {len(hist_df)} days.")
    
    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        metrics = st.session_state.accuracy_metrics
        
        # Accuracy metrics
        st.subheader("🎯 Accuracy Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        overall = metrics.get('overall', {})
        
        with col1:
            acc = overall.get('customer_accuracy', 0)
            st.metric("Customer Accuracy", f"{acc:.1f}%")
        
        with col2:
            acc = overall.get('sales_accuracy', 0)
            st.metric("Sales Accuracy", f"{acc:.1f}%")
        
        with col3:
            bias = overall.get('customer_bias', 0)
            bias_dir = "Under" if bias < 0 else "Over"
            st.metric("Customer Bias", f"{bias_dir} {abs(bias):.1f}%")
        
        with col4:
            st.metric("Predictions Tested", f"{overall.get('n_predictions', 0)}")
        
        # By horizon
        st.subheader("📏 Accuracy by Forecast Horizon")
        
        horizon_data = []
        for h in [1, 3, 7, 14]:
            h_metrics = metrics.get(f'horizon_{h}d', {})
            if h_metrics:
                horizon_data.append({
                    'Horizon': f'{h} day{"s" if h > 1 else ""}',
                    'Customer Accuracy': f"{h_metrics.get('customer_accuracy', 0):.1f}%",
                    'Sales Accuracy': f"{h_metrics.get('sales_accuracy', 0):.1f}%",
                    'Bias': f"{h_metrics.get('customer_bias', 0):+.1f}%",
                    'Samples': h_metrics.get('n_predictions', 0)
                })
        
        if horizon_data:
            st.dataframe(pd.DataFrame(horizon_data).set_index('Horizon'), use_container_width=True)
        
        # Charts
        tab1, tab2 = st.tabs(["Customers", "Sales"])
        
        with tab1:
            fig = create_backtest_chart(results, 'customers')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_backtest_chart(results, 'sales')
            st.plotly_chart(fig, use_container_width=True)
        
        # By day of week
        st.subheader("📅 Accuracy by Day of Week")
        
        dow_metrics = metrics.get('by_dow', {})
        if dow_metrics:
            dow_df = pd.DataFrame([
                {
                    'Day': day,
                    'Accuracy': f"{data['accuracy']:.1f}%",
                    'Bias': f"{data['bias']:+.1f}%",
                    'Samples': data['n']
                }
                for day, data in dow_metrics.items()
            ])
            st.dataframe(dow_df.set_index('Day'), use_container_width=True)
    
    else:
        st.info("👈 Click 'Run Backtest' to evaluate model accuracy")


# === PAGE: Trend Analysis ===
elif page == "📈 Trend Analysis":
    st.header("📈 Trend & Contextual Analysis")
    
    if not hist_df.empty:
        # 8-Week Recent Trend
        st.subheader("📊 8-Week Recent Performance")
        st.caption("Recent weeks get MORE weight in predictions (Week 1 = 8x, Week 8 = 1x)")
        
        fig = create_recent_trend_chart(hist_df, weeks=8)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Calculate recent trend features for display
        target_date = hist_df['date'].max() + timedelta(days=1)
        recent_features = calculate_recent_trend_features(hist_df, target_date, weeks=8)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            val = recent_features.get('recent_weighted_cust_mean', 0)
            st.metric("Weighted Avg Customers", f"{val:.0f}" if not np.isnan(val) else "N/A")
        
        with col2:
            val = recent_features.get('recent_momentum_2w', 1)
            if not np.isnan(val):
                delta = f"{(val - 1) * 100:+.1f}%"
                st.metric("2-Week Momentum", f"{val:.2f}", delta=delta)
            else:
                st.metric("2-Week Momentum", "N/A")
        
        with col3:
            val = recent_features.get('recent_momentum_4w', 1)
            if not np.isnan(val):
                delta = f"{(val - 1) * 100:+.1f}%"
                st.metric("4-Week Momentum", f"{val:.2f}", delta=delta)
            else:
                st.metric("4-Week Momentum", "N/A")
        
        with col4:
            val = recent_features.get('recent_volatility', 0)
            st.metric("Volatility (CV)", f"{val:.2%}" if not np.isnan(val) else "N/A")
        
        # SDLY Contextual Window
        st.divider()
        st.subheader("🗓️ SDLY Contextual Window")
        st.caption("Same Day Last Year ± 14 days - captures seasonal patterns")
        
        # Let user select a date to analyze
        forecast_date = st.date_input(
            "Select date to analyze",
            value=hist_df['date'].max() + timedelta(days=1),
            min_value=hist_df['date'].min() + timedelta(days=365)
        )
        
        fig = create_contextual_window_chart(hist_df, pd.to_datetime(forecast_date))
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Contextual features
        ctx_features = calculate_contextual_window_features(hist_df, pd.to_datetime(forecast_date))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            val = ctx_features.get('sdly_customers', np.nan)
            st.metric("SDLY Customers", f"{val:.0f}" if not np.isnan(val) else "N/A")
        
        with col2:
            val = ctx_features.get('sdly_window_cust_mean', np.nan)
            st.metric("Window Average", f"{val:.0f}" if not np.isnan(val) else "N/A")
        
        with col3:
            val = ctx_features.get('sdly_momentum', np.nan)
            if not np.isnan(val):
                delta = f"{(val - 1) * 100:+.1f}%"
                st.metric("Window Momentum", f"{val:.2f}", delta=delta)
            else:
                st.metric("Window Momentum", "N/A")
        
        # Trend analysis
        st.markdown("**Trend Analysis:**")
        trend_before = ctx_features.get('sdly_trend_before', 0)
        trend_after = ctx_features.get('sdly_trend_after', 0)
        
        if not np.isnan(trend_before) and not np.isnan(trend_after):
            st.markdown(f"""
            - **14 days BEFORE SDLY**: {"📈 Increasing" if trend_before > 0.02 else "📉 Decreasing" if trend_before < -0.02 else "➡️ Stable"} ({trend_before*100:+.1f}%)
            - **14 days AFTER SDLY**: {"📈 Increasing" if trend_after > 0.02 else "📉 Decreasing" if trend_after < -0.02 else "➡️ Stable"} ({trend_after*100:+.1f}%)
            """)
    else:
        st.info("No historical data available for trend analysis")


# Footer
st.divider()
st.caption("AI Sales Forecaster v10.0 | Self-Correcting Ensemble with Contextual Window Analysis")
