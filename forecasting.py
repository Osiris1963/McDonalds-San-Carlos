# forecasting.py - COGNITIVE FORECASTING ENGINE v11.0
# Key enhancements:
# 1. Self-Correcting AI - learns from past forecast errors
# 2. Contextual Window Analysis (SDLY ± 14 days)  
# 3. 8-Week Weighted Recent Trends (more weight to recent data)
# 4. Blended prediction combining historical patterns + current trends
# 5. Automatic Learning - calibrates when new data arrives
# 6. Intelligent Anomaly Handling
# 7. Multi-Scenario Forecasting
# 8. Confidence Scoring & Explanations

import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from datetime import timedelta
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from data_processing import (
    create_advanced_features, 
    prepare_data_for_prophet,
    get_feature_list,
    calculate_historical_multipliers,
    detect_special_dates,
    load_forecast_errors,
    calculate_contextual_window_features,
    calculate_recent_trend_features,
    calculate_yoy_comparison
)

# Import intelligent components (with fallback if not available)
try:
    from intelligent_engine import (
        CognitiveForecaster,
        AnomalyIntelligence,
        RegimeDetector,
        ExplanationGenerator,
        ConfidenceScorer,
        MultiScenarioForecaster
    )
    from auto_learning import (
        AutoLearningSystem,
        RealTimeLearner,
        AdaptiveWeightManager
    )
    INTELLIGENT_MODE = True
except ImportError:
    INTELLIGENT_MODE = False
    print("Warning: Intelligent components not available. Running in basic mode.")


class SelfCorrectingForecaster:
    """
    Self-Correcting Forecaster that learns from past mistakes.
    
    How it works:
    1. Loads historical forecast errors from Firestore
    2. Identifies systematic biases (e.g., always under-predicting Saturdays)
    3. Applies correction factors to new predictions
    """
    
    def __init__(self, db_client=None):
        self.db_client = db_client
        self.correction_factors = {}
        self.error_patterns = {}
        self.is_calibrated = False
        
    def calibrate(self):
        """
        Analyze past forecast errors and learn correction patterns.
        """
        if self.db_client is None:
            print("No database client - skipping self-correction calibration")
            return self
        
        errors_df = load_forecast_errors(self.db_client)
        
        if errors_df.empty or len(errors_df) < 7:
            print(f"Insufficient error history ({len(errors_df) if not errors_df.empty else 0} records) - need at least 7")
            return self
        
        print(f"Calibrating self-correction from {len(errors_df)} historical predictions...")
        
        # === LEARN SYSTEMATIC BIASES ===
        
        # 1. Day-of-week bias
        dow_errors = errors_df.groupby('dayofweek').agg({
            'customer_pct_error': 'mean',
            'customer_error': 'mean'
        }).to_dict()
        
        self.correction_factors['dow'] = {}
        for dow in range(7):
            if dow in dow_errors['customer_pct_error']:
                # If we consistently under-predict (negative error), correction > 1
                # If we consistently over-predict (positive error), correction < 1
                bias = dow_errors['customer_pct_error'].get(dow, 0)
                self.correction_factors['dow'][dow] = 1 + (bias / 100)
        
        # 2. Payday bias
        payday_errors = errors_df.groupby('is_payday')['customer_pct_error'].mean()
        if 1 in payday_errors.index:
            self.correction_factors['payday'] = 1 + (payday_errors[1] / 100)
        else:
            self.correction_factors['payday'] = 1.0
        
        # 3. Month bias
        month_errors = errors_df.groupby('month')['customer_pct_error'].mean().to_dict()
        self.correction_factors['month'] = {
            m: 1 + (err / 100) for m, err in month_errors.items()
        }
        
        # 4. Overall bias (systematic over/under prediction)
        overall_bias = errors_df['customer_pct_error'].mean()
        self.correction_factors['overall'] = 1 + (overall_bias / 100)
        
        # 5. Recent trend in errors (are we getting worse or better?)
        if len(errors_df) >= 14:
            recent_errors = errors_df.tail(7)['customer_pct_error'].mean()
            older_errors = errors_df.head(len(errors_df) - 7)['customer_pct_error'].mean()
            self.correction_factors['trend'] = recent_errors - older_errors
        else:
            self.correction_factors['trend'] = 0
        
        # Store error patterns for diagnostics
        self.error_patterns = {
            'by_dow': dow_errors,
            'by_month': month_errors,
            'overall_mape': abs(errors_df['customer_pct_error']).mean(),
            'recent_mape': abs(errors_df.tail(7)['customer_pct_error']).mean() if len(errors_df) >= 7 else None,
            'n_samples': len(errors_df)
        }
        
        self.is_calibrated = True
        print(f"Self-correction calibrated. Overall bias: {overall_bias:.1f}%")
        
        return self
    
    def apply_correction(self, base_prediction, target_date):
        """
        Apply learned correction factors to a base prediction.
        """
        if not self.is_calibrated:
            return base_prediction
        
        correction = 1.0
        
        # Apply day-of-week correction
        dow = target_date.weekday()
        if dow in self.correction_factors.get('dow', {}):
            dow_corr = self.correction_factors['dow'][dow]
            # Limit correction to ±20% to avoid wild swings
            dow_corr = np.clip(dow_corr, 0.8, 1.2)
            correction *= dow_corr
        
        # Apply month correction
        month = target_date.month
        if month in self.correction_factors.get('month', {}):
            month_corr = self.correction_factors['month'][month]
            month_corr = np.clip(month_corr, 0.9, 1.1)
            correction *= month_corr
        
        # Apply payday correction
        if target_date.day in [14, 15, 16, 29, 30, 31, 1, 2]:
            payday_corr = self.correction_factors.get('payday', 1.0)
            payday_corr = np.clip(payday_corr, 0.9, 1.15)
            correction *= payday_corr
        
        # Apply overall bias correction (weighted less)
        overall_corr = self.correction_factors.get('overall', 1.0)
        overall_corr = np.clip(overall_corr, 0.95, 1.05)
        correction *= (overall_corr ** 0.5)  # Square root to reduce impact
        
        return base_prediction * correction
    
    def get_diagnostics(self):
        """Return learned error patterns for display."""
        return {
            'correction_factors': self.correction_factors,
            'error_patterns': self.error_patterns,
            'is_calibrated': self.is_calibrated
        }


class EnhancedCustomerForecaster:
    """
    Enhanced Customer Forecaster with:
    1. Contextual Window Analysis (SDLY ± 14 days)
    2. 8-Week Weighted Recent Trends
    3. Self-Correction from past errors
    4. Blended prediction strategy
    """
    
    def __init__(self, db_client=None, n_estimators=500, learning_rate=0.02, max_depth=7):
        self.model = None
        self.feature_cols = None
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.multipliers = {}
        self.dow_medians = {}
        self.db_client = db_client
        
        # Self-correction component
        self.self_corrector = SelfCorrectingForecaster(db_client)
        
        # Blending weights (adjusted based on data availability)
        self.blend_weights = {
            'model': 0.50,           # LightGBM model prediction
            'sdly_contextual': 0.25, # SDLY with contextual window
            'recent_trend': 0.25     # 8-week weighted recent trend
        }
        
    def fit(self, historical_df, events_df=None):
        """Train the enhanced model."""
        
        # Step 1: Calibrate self-correction from past errors
        self.self_corrector.calibrate()
        
        # Step 2: Create features with contextual window analysis
        print("Creating advanced features with contextual window analysis...")
        train_data = create_advanced_features(historical_df, events_df)
        
        # Step 3: Calculate historical multipliers
        self.multipliers = calculate_historical_multipliers(historical_df, events_df)
        
        # Step 4: Store day-of-week medians for baseline
        self.dow_medians = train_data.groupby('dayofweek')['customers'].median().to_dict()
        
        # Step 5: Define features
        self.feature_cols = get_feature_list(include_contextual=True, include_recent=True)
        self.feature_cols = [f for f in self.feature_cols if f in train_data.columns]
        
        # Step 6: Prepare training data
        train_clean = train_data.dropna(subset=['customers'])
        
        # Fill remaining NaN with 0 for optional features
        for col in self.feature_cols:
            if col in train_clean.columns:
                train_clean[col] = train_clean[col].fillna(0)
        
        # Filter to rows that have the core features
        core_features = ['dayofweek', 'month', 'customers_lag_1']
        core_available = [f for f in core_features if f in train_clean.columns]
        train_clean = train_clean.dropna(subset=core_available)
        
        if len(train_clean) < 60:
            raise ValueError(f"Insufficient training data: {len(train_clean)} rows (need 60+)")
        
        X = train_clean[self.feature_cols].fillna(0)
        y = train_clean['customers']
        
        # Step 7: Train LightGBM with enhanced parameters
        print("Training LightGBM model...")
        self.model = lgb.LGBMRegressor(
            objective='tweedie',
            tweedie_variance_power=1.5,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=63,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
            random_state=42,
            importance_type='gain'
        )
        
        self.model.fit(X, y)
        
        # Store for recursive prediction
        self.last_train_data = train_data.copy()
        self.historical_df = historical_df.copy()
        self.events_df = events_df
        
        print(f"Model trained on {len(train_clean)} samples with {len(self.feature_cols)} features")
        
        return self
    
    def predict_recursive(self, periods=15):
        """
        Generate forecasts using BLENDED recursive prediction.
        
        For each day:
        1. Get LightGBM model prediction
        2. Get SDLY contextual prediction (last year same period ± 14 days)
        3. Get recent trend prediction (8-week weighted)
        4. Blend all three
        5. Apply self-correction
        6. Apply special date multipliers
        """
        predictions = []
        
        working_df = self.historical_df.copy()
        last_date = working_df['date'].max()
        
        for h in range(1, periods + 1):
            target_date = last_date + timedelta(days=h)
            
            # === COMPONENT 1: LightGBM Model Prediction ===
            new_row = pd.DataFrame({
                'date': [target_date],
                'customers': [np.nan],
                'sales': [np.nan],
                'atv': [np.nan]
            })
            
            temp_df = pd.concat([working_df, new_row], ignore_index=True)
            featured_df = create_advanced_features(temp_df, self.events_df)
            
            target_row = featured_df[featured_df['date'] == target_date].iloc[0]
            
            X_pred = pd.DataFrame([target_row[self.feature_cols].fillna(0)])
            model_pred = max(0, self.model.predict(X_pred)[0])
            
            # === COMPONENT 2: SDLY Contextual Prediction ===
            ctx_features = calculate_contextual_window_features(working_df, target_date)
            
            # Use SDLY with YoY growth adjustment
            yoy = calculate_yoy_comparison(working_df, target_date)
            
            if not np.isnan(ctx_features['sdly_customers']) and ctx_features['sdly_customers'] > 0:
                # Adjust SDLY by YoY growth rate
                sdly_pred = ctx_features['sdly_customers'] * yoy['yoy_customer_growth']
                
                # Also consider the trend from the contextual window
                if ctx_features['sdly_momentum'] > 0:
                    sdly_pred *= (1 + ctx_features['sdly_trend_after'] * 0.3)
            else:
                # Fallback to window mean
                sdly_pred = ctx_features.get('sdly_window_cust_mean', model_pred)
                if np.isnan(sdly_pred):
                    sdly_pred = model_pred
            
            # === COMPONENT 3: Recent Trend Prediction ===
            recent_features = calculate_recent_trend_features(working_df, target_date, weeks=8)
            
            recent_base = recent_features.get('recent_weighted_cust_mean', model_pred)
            if np.isnan(recent_base):
                recent_base = model_pred
            
            # Apply day-of-week factor from recent data
            dow_factor = recent_features.get('recent_dow_factor', 1.0)
            if np.isnan(dow_factor):
                dow_factor = 1.0
            
            # Apply momentum
            momentum = recent_features.get('recent_momentum_2w', 1.0)
            if np.isnan(momentum):
                momentum = 1.0
            momentum = np.clip(momentum, 0.85, 1.15)
            
            recent_pred = recent_base * dow_factor * momentum
            
            # === BLEND THE THREE PREDICTIONS ===
            w_model = self.blend_weights['model']
            w_sdly = self.blend_weights['sdly_contextual']
            w_recent = self.blend_weights['recent_trend']
            
            # If SDLY is unavailable, redistribute weight
            if np.isnan(sdly_pred) or sdly_pred <= 0:
                sdly_pred = model_pred
                w_model += w_sdly * 0.5
                w_recent += w_sdly * 0.5
                w_sdly = 0
            
            blended_pred = (
                w_model * model_pred +
                w_sdly * sdly_pred +
                w_recent * recent_pred
            )
            
            # === APPLY SELF-CORRECTION ===
            corrected_pred = self.self_corrector.apply_correction(blended_pred, target_date)
            
            # === APPLY SPECIAL DATE MULTIPLIERS ===
            final_pred = self._apply_multipliers(corrected_pred, target_date, target_row)
            
            # Ensure non-negative
            final_pred = max(0, final_pred)
            
            predictions.append({
                'ds': target_date,
                'forecast_customers': int(round(final_pred)),
                'model_prediction': model_pred,
                'sdly_prediction': sdly_pred,
                'recent_prediction': recent_pred,
                'blended_prediction': blended_pred,
                'corrected_prediction': corrected_pred,
                'yoy_growth': yoy['yoy_customer_growth'],
                'recent_momentum': momentum
            })
            
            # Update working_df with prediction for next iteration
            working_df = pd.concat([
                working_df,
                pd.DataFrame({
                    'date': [target_date],
                    'customers': [final_pred],
                    'sales': [np.nan],
                    'atv': [np.nan]
                })
            ], ignore_index=True)
        
        return pd.DataFrame(predictions)
    
    def _apply_multipliers(self, base_pred, date, features_row):
        """Apply learned multipliers for special dates."""
        multiplier = 1.0
        
        # New Year
        if date.month == 1 and date.day <= 2:
            multiplier *= self.multipliers.get('new_year', 1.3)
        
        # Christmas
        if date.month == 12 and date.day >= 20:
            multiplier *= self.multipliers.get('christmas', 1.15)
        
        # Event day
        if features_row.get('is_event', 0) == 1:
            multiplier *= self.multipliers.get('event', 1.1)
        
        return base_pred * multiplier
    
    def get_feature_importance(self):
        """Return feature importance."""
        if self.model is None:
            return None
        
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def get_blend_diagnostics(self):
        """Return blending and correction diagnostics."""
        return {
            'blend_weights': self.blend_weights,
            'multipliers': self.multipliers,
            'self_correction': self.self_corrector.get_diagnostics()
        }


class ATVForecaster:
    """Prophet-based forecaster for Average Transaction Value."""
    
    def __init__(self):
        self.model = None
        self.regressors = None
    
    def fit(self, historical_df, events_df=None):
        df_p, self.regressors = prepare_data_for_prophet(historical_df, events_df)
        df_p = df_p[df_p['y'] > 0].copy()
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            interval_width=0.8
        )
        
        for reg in self.regressors:
            self.model.add_regressor(reg, mode='multiplicative')
        
        self.model.fit(df_p)
        self.events_df = events_df
        
        return self
    
    def predict(self, periods=15):
        future = self.model.make_future_dataframe(periods=periods)
        
        future['is_payday'] = future['ds'].dt.day.isin([14, 15, 16, 29, 30, 31, 1, 2]).astype(int)
        future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
        future['is_friday'] = (future['ds'].dt.dayofweek == 4).astype(int)
        future['is_new_year'] = ((future['ds'].dt.month == 1) & (future['ds'].dt.day <= 2)).astype(int)
        future['is_christmas'] = ((future['ds'].dt.month == 12) & (future['ds'].dt.day >= 20)).astype(int)
        
        if self.events_df is not None and not self.events_df.empty:
            ev_dates = pd.to_datetime(self.events_df['date']).dt.normalize()
            future['is_event'] = future['ds'].isin(ev_dates).astype(int)
        else:
            future['is_event'] = 0
        
        forecast = self.model.predict(future)
        
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
        result.columns = ['ds', 'forecast_atv', 'atv_lower', 'atv_upper']
        
        return result


class EnsembleForecaster:
    """
    Main forecasting class combining all components.
    v11.0: Now with Cognitive Intelligence
    """
    
    def __init__(self, db_client=None):
        self.customer_forecaster = EnhancedCustomerForecaster(db_client)
        self.atv_forecaster = ATVForecaster()
        self.db_client = db_client
        self.is_fitted = False
        
        # Initialize intelligent components if available
        if INTELLIGENT_MODE and db_client:
            self.cognitive = CognitiveForecaster(db_client)
            self.auto_learner = AutoLearningSystem(db_client)
            self.weight_manager = AdaptiveWeightManager(db_client)
            self.realtime_learner = RealTimeLearner(db_client)
        else:
            self.cognitive = None
            self.auto_learner = None
            self.weight_manager = None
            self.realtime_learner = None
    
    def fit(self, historical_df, events_df=None):
        print("=" * 60)
        print("COGNITIVE FORECASTING ENGINE v11.0")
        print("=" * 60)
        
        # Step 0: Intelligent pre-analysis (if available)
        if self.cognitive:
            print("\n[0/3] Running Cognitive Analysis...")
            self.analysis = self.cognitive.analyze_and_prepare(historical_df, events_df)
            
            # Check for regime changes
            if self.analysis['regime'].get('regime_change'):
                print("⚠️  REGIME CHANGE DETECTED!")
                print(f"    {self.analysis['regime'].get('recommendation')}")
            
            # Check for anomalies
            n_anomalies = len(self.analysis['anomalies'])
            if n_anomalies > 0:
                print(f"📊 Detected {n_anomalies} anomalies in historical data")
            
            # Check if auto-learning should run
            if self.analysis['learning_status'].get('should_recalibrate'):
                print(f"🔄 Auto-recalibration triggered: {self.analysis['learning_status'].get('reason')}")
        
        # Step 1: Load adaptive weights if available
        if self.weight_manager:
            adaptive_weights = self.weight_manager.get_current_weights()
            self.customer_forecaster.blend_weights = adaptive_weights
            print(f"\n[1/3] Using adaptive blend weights: Model={adaptive_weights['model']:.0%}, SDLY={adaptive_weights['sdly_contextual']:.0%}, Recent={adaptive_weights['recent_trend']:.0%}")
        
        # Step 2: Train Customer Forecaster
        print("\n[2/3] Training Customer Forecaster with Self-Correction...")
        self.customer_forecaster.fit(historical_df, events_df)
        
        # Step 3: Train ATV Forecaster
        print("\n[3/3] Training ATV Forecaster...")
        self.atv_forecaster.fit(historical_df, events_df)
        
        self.historical_df = historical_df
        self.events_df = events_df
        self.is_fitted = True
        
        print("\n✅ Training complete!")
        return self
    
    def predict(self, periods=15, include_scenarios=True, include_explanations=True):
        """
        Generate intelligent forecast with optional scenarios and explanations.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get base forecasts
        cust_forecast = self.customer_forecaster.predict_recursive(periods)
        atv_forecast = self.atv_forecaster.predict(periods)
        
        # Merge
        result = pd.merge(cust_forecast, atv_forecast, on='ds')
        
        # Calculate sales
        result['forecast_sales'] = result['forecast_customers'] * result['forecast_atv']
        result['sales_lower'] = result['forecast_customers'] * result['atv_lower']
        result['sales_upper'] = result['forecast_customers'] * result['atv_upper']
        
        # Add intelligent enhancements if available
        if self.cognitive and include_scenarios:
            enhanced, regime_info = self.cognitive.generate_intelligent_forecast(
                result,
                self.historical_df,
                self.customer_forecaster.self_corrector.correction_factors if hasattr(self.customer_forecaster, 'self_corrector') else None,
                self.customer_forecaster.get_feature_importance()
            )
            
            # Add scenarios and confidence to result
            for i, row in enumerate(enhanced):
                result.loc[i, 'confidence_score'] = row['confidence']['score']
                result.loc[i, 'confidence_grade'] = row['confidence']['grade']
                result.loc[i, 'optimistic_forecast'] = row['scenarios'][1]['forecast']
                result.loc[i, 'pessimistic_forecast'] = row['scenarios'][2]['forecast']
                
                if include_explanations:
                    result.loc[i, 'explanation'] = row['explanation']
            
            self.regime_info = regime_info
        
        return result
    
    def run_auto_learning(self):
        """Trigger automatic learning cycle."""
        if not self.auto_learner or not self.is_fitted:
            return {'status': 'not_available'}
        
        # Load forecast log
        try:
            forecast_docs = self.db_client.collection('forecast_log').stream()
            forecasts = []
            for doc in forecast_docs:
                record = doc.to_dict()
                record['date'] = doc.id
                forecasts.append(record)
            forecast_df = pd.DataFrame(forecasts)
            
            if forecast_df.empty:
                return {'status': 'no_forecast_history'}
            
            # Run learning cycle
            result = self.auto_learner.run_learning_cycle(self.historical_df, forecast_df)
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def record_actual(self, date, actual_customers, actual_sales, actual_atv):
        """Record actual values for real-time learning."""
        if self.realtime_learner:
            return self.realtime_learner.record_actual(date, actual_customers, actual_sales, actual_atv)
        return {'status': 'not_available'}
    
    def get_diagnostics(self):
        diag = {
            'feature_importance': self.customer_forecaster.get_feature_importance(),
            'blend_diagnostics': self.customer_forecaster.get_blend_diagnostics(),
            'multipliers': self.customer_forecaster.multipliers,
            'intelligent_mode': INTELLIGENT_MODE
        }
        
        if hasattr(self, 'analysis'):
            diag['cognitive_analysis'] = {
                'data_quality': self.analysis['data_quality'],
                'regime_info': self.analysis['regime'],
                'recommendations': self.analysis['recommendations'],
                'n_anomalies': len(self.analysis['anomalies'])
            }
        
        if hasattr(self, 'regime_info'):
            diag['regime_info'] = self.regime_info
        
        return diag
    
    def generate_briefing(self, forecast_df):
        """Generate executive briefing."""
        if self.cognitive:
            anomalies = self.analysis.get('anomalies', pd.DataFrame()) if hasattr(self, 'analysis') else pd.DataFrame()
            regime = self.regime_info if hasattr(self, 'regime_info') else {}
            return self.cognitive.explainer.generate_daily_briefing(forecast_df, anomalies, regime)
        return "Intelligent briefing not available in basic mode."


# === BACKTESTING ===

def backtest_model(historical_df, events_df, db_client=None, test_days=30, step_size=7):
    """
    Walk-forward backtesting to evaluate model performance.
    """
    results = []
    
    max_date = historical_df['date'].max()
    min_test_date = max_date - timedelta(days=test_days)
    
    earliest_train_end = historical_df['date'].min() + timedelta(days=365)
    
    if min_test_date < earliest_train_end:
        min_test_date = earliest_train_end
    
    current_test_start = min_test_date
    iteration = 0
    
    while current_test_start < max_date:
        iteration += 1
        
        train_df = historical_df[historical_df['date'] < current_test_start].copy()
        
        days_available = (max_date - current_test_start).days
        forecast_horizon = min(15, days_available)
        
        if forecast_horizon < 1:
            break
        
        print(f"  Backtest iteration {iteration}: Training up to {current_test_start.strftime('%Y-%m-%d')}")
        
        try:
            forecaster = EnsembleForecaster(db_client=None)
            forecaster.fit(train_df, events_df)
            predictions = forecaster.predict(periods=forecast_horizon)
            
            test_period = historical_df[
                (historical_df['date'] > current_test_start) & 
                (historical_df['date'] <= current_test_start + timedelta(days=forecast_horizon))
            ].copy()
            
            for _, pred_row in predictions.iterrows():
                actual_row = test_period[test_period['date'] == pred_row['ds'].normalize()]
                
                if len(actual_row) > 0:
                    actual_row = actual_row.iloc[0]
                    results.append({
                        'date': pred_row['ds'],
                        'horizon': (pred_row['ds'] - current_test_start).days,
                        'actual_customers': actual_row['customers'],
                        'predicted_customers': pred_row['forecast_customers'],
                        'actual_sales': actual_row['sales'],
                        'predicted_sales': pred_row['forecast_sales'],
                        'actual_atv': actual_row['atv'],
                        'predicted_atv': pred_row['forecast_atv'],
                        'model_pred': pred_row.get('model_prediction', np.nan),
                        'sdly_pred': pred_row.get('sdly_prediction', np.nan),
                        'recent_pred': pred_row.get('recent_prediction', np.nan)
                    })
        except Exception as e:
            print(f"    Warning: Backtest iteration failed: {e}")
        
        current_test_start += timedelta(days=step_size)
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        results_df['customer_error'] = results_df['actual_customers'] - results_df['predicted_customers']
        results_df['customer_pct_error'] = abs(results_df['customer_error']) / results_df['actual_customers'] * 100
        
        results_df['sales_error'] = results_df['actual_sales'] - results_df['predicted_sales']
        results_df['sales_pct_error'] = abs(results_df['sales_error']) / results_df['actual_sales'] * 100
        
        results_df['customer_signed_pct_error'] = results_df['customer_error'] / results_df['actual_customers'] * 100
        results_df['sales_signed_pct_error'] = results_df['sales_error'] / results_df['actual_sales'] * 100
    
    return results_df


def calculate_accuracy_metrics(backtest_results):
    """Calculate comprehensive accuracy metrics from backtest results."""
    if backtest_results.empty:
        return {}
    
    metrics = {
        'overall': {
            'customer_mape': backtest_results['customer_pct_error'].mean(),
            'customer_accuracy': 100 - backtest_results['customer_pct_error'].mean(),
            'sales_mape': backtest_results['sales_pct_error'].mean(),
            'sales_accuracy': 100 - backtest_results['sales_pct_error'].mean(),
            'customer_bias': backtest_results['customer_signed_pct_error'].mean(),
            'sales_bias': backtest_results['sales_signed_pct_error'].mean(),
            'n_predictions': len(backtest_results)
        }
    }
    
    for horizon in [1, 3, 7, 14]:
        horizon_data = backtest_results[backtest_results['horizon'] <= horizon]
        if len(horizon_data) > 0:
            metrics[f'horizon_{horizon}d'] = {
                'customer_accuracy': 100 - horizon_data['customer_pct_error'].mean(),
                'sales_accuracy': 100 - horizon_data['sales_pct_error'].mean(),
                'customer_bias': horizon_data['customer_signed_pct_error'].mean(),
                'n_predictions': len(horizon_data)
            }
    
    backtest_results['dayofweek'] = pd.to_datetime(backtest_results['date']).dt.dayofweek
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    metrics['by_dow'] = {}
    for dow in range(7):
        dow_data = backtest_results[backtest_results['dayofweek'] == dow]
        if len(dow_data) > 0:
            metrics['by_dow'][dow_names[dow]] = {
                'accuracy': 100 - dow_data['customer_pct_error'].mean(),
                'bias': dow_data['customer_signed_pct_error'].mean(),
                'n': len(dow_data)
            }
    
    return metrics


# === LEGACY API COMPATIBILITY ===

def generate_customer_forecast(historical_df, events_df, periods=15, db_client=None):
    """Legacy API: Returns customer forecast."""
    forecaster = EnhancedCustomerForecaster(db_client)
    forecaster.fit(historical_df, events_df)
    result = forecaster.predict_recursive(periods)
    return result[['ds', 'forecast_customers']], forecaster


def generate_atv_forecast(historical_df, events_df, periods=15):
    """Legacy API: Returns ATV forecast."""
    forecaster = ATVForecaster()
    forecaster.fit(historical_df, events_df)
    result = forecaster.predict(periods)
    return result[['ds', 'forecast_atv']], forecaster.model


def generate_full_forecast(historical_df, events_df, periods=15, db_client=None):
    """Main API: Returns complete forecast with all diagnostics."""
    forecaster = EnsembleForecaster(db_client)
    forecaster.fit(historical_df, events_df)
    result = forecaster.predict(periods)
    diagnostics = forecaster.get_diagnostics()
    return result, forecaster, diagnostics
