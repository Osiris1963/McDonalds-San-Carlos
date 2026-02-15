# forecasting.py - COGNITIVE FORECASTING ENGINE v12.2 (SDLY-ANCHORED)
# 
# KEY PHILOSOPHY CHANGE (v12.2):
#   Old: blend(ML, SDLY, Recent) → dampen → conservative → min() = TOO LOW
#   New: SDLY × measured_growth × DOW_adj + ML_residual(±15%) = REALITY-BASED
#
# The forecast is now ANCHORED to what actually happened last year,
# adjusted by how much better/worse the business is ACTUALLY doing.
# ML model is a NUDGE (±15%), not the driver.
#
# Previous enhancements preserved:
# 1. Self-Correcting AI - learns from past forecast errors
# 2. Contextual Window Analysis (SDLY ± 14 days)
# 3. Fast single-row feature computation (no O(N²))
# 4. NaN propagation fixed
# 5. Early stopping on LightGBM
# 6. Proper confidence intervals from residual distribution
# 7. Philippine holiday calendar
# 8. Anomaly detection wired into training

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
    calculate_yoy_comparison,
    compute_single_row_features
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
except ImportError as e:
    INTELLIGENT_MODE = False
    print(f"⚠️ Intelligent components not available: {e}")

# Import trend intelligence (v12.0)
try:
    from trend_intelligence import (
        TrendAnalyzer,
        TrendAwareBlender,
        MomentumDampener,
        ConservativeEstimator,
        apply_trend_intelligence
    )
    TREND_AWARE_MODE = True
except ImportError as e:
    TREND_AWARE_MODE = False
    print(f"⚠️ Trend intelligence not available: {e}")


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
        FIXED: Clips the FINAL compound correction to prevent extreme adjustments.
        """
        if not self.is_calibrated:
            return base_prediction
        
        correction = 1.0
        
        # Apply day-of-week correction
        dow = target_date.weekday()
        if dow in self.correction_factors.get('dow', {}):
            dow_corr = self.correction_factors['dow'][dow]
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
        correction *= (overall_corr ** 0.5)
        
        # FIXED: Clip the FINAL compound correction to prevent extreme adjustments
        correction = np.clip(correction, 0.80, 1.25)
        
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
        
        # Step 7: Train LightGBM with early stopping on time-series validation
        print("Training LightGBM model with early stopping...")
        
        # Time-series split: use last 20% as validation (preserving temporal order)
        split_idx = int(len(X) * 0.80)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
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
        
        # Early stopping prevents overfitting
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Store validation MAPE for diagnostics
        val_pred = self.model.predict(X_val)
        val_mape = np.mean(np.abs(y_val - val_pred) / y_val.replace(0, np.nan).dropna()) * 100
        self.validation_mape = val_mape
        print(f"  Validation MAPE: {val_mape:.1f}%  (used {self.model.best_iteration_} trees)")
        
        # Store for recursive prediction
        self.last_train_data = train_data.copy()
        self.historical_df = historical_df.copy()
        self.events_df = events_df
        
        print(f"Model trained on {len(train_clean)} samples with {len(self.feature_cols)} features")
        
        return self
    
    def predict_recursive(self, periods=15):
        """
        SDLY-ANCHORED FORECASTING ENGINE v12.2
        
        Philosophy change: SDLY is the ANCHOR, not one of three equal votes.
        
        Old approach (broken):
          blend(ML, SDLY, Recent) → dampen → conservative → min() = TOO LOW
          
        New approach:
          1. SDLY Base: What happened on this exact day last year?
          2. Measured Growth: How much better/worse are we vs last year? (from ACTUAL data)
          3. Growth-Adjusted SDLY = SDLY × measured_growth (THE ANCHOR)
          4. DOW Fine-Tune: Adjust for day-of-week patterns in recent weeks
          5. ML Residual: Model says "this specific day should be ±X% vs pattern" (CAPPED at ±15%)
          6. Final = Anchored prediction + capped ML correction
          
        This GUARANTEES:
        - If recent actuals are 8% above last year → forecast is ~8% above last year
        - ML model can only nudge ±15% for specific days (payday, events, etc.)
        - No cascading dampeners can drag forecast below reality
        """
        predictions = []
        
        working_df = self.historical_df.copy()
        last_date = working_df['date'].max()
        
        # === PRE-COMPUTE: Measured YoY Growth at Multiple Windows ===
        growth_rates = self._measure_yoy_growth(working_df)
        print(f"\n📊 MEASURED YOY GROWTH RATES:")
        print(f"   7-day:  {growth_rates['7d']*100:+.1f}%")
        print(f"   14-day: {growth_rates['14d']*100:+.1f}%")
        print(f"   30-day: {growth_rates['30d']*100:+.1f}%")
        print(f"   Composite: {growth_rates['composite']*100:+.1f}%")
        
        # === PRE-COMPUTE: DOW patterns from recent 4 weeks ===
        dow_patterns = self._compute_dow_patterns(working_df)
        
        # === PRE-COMPUTE: Recent ATV for sales estimation ===
        recent_atv_median = working_df.tail(30)['atv'].median()
        if np.isnan(recent_atv_median) or recent_atv_median <= 0:
            recent_atv_median = working_df['atv'].median()
        
        # === PRE-COMPUTE: Residual std for confidence intervals ===
        customer_residual_std = self._estimate_residual_std(working_df)
        
        for h in range(1, periods + 1):
            target_date = last_date + timedelta(days=h)
            target_dow = target_date.weekday()
            
            # ============================================================
            # STEP 1: SDLY BASE (Same Day Last Year - the ANCHOR)
            # ============================================================
            sdly_date = target_date - timedelta(days=364)  # DOW-aligned
            sdly_row = working_df[working_df['date'] == sdly_date]
            
            if len(sdly_row) > 0:
                sdly_exact = sdly_row['customers'].values[0]
            else:
                # Fallback: average of same DOW in ±7 day window around SDLY
                sdly_window_start = sdly_date - timedelta(days=7)
                sdly_window_end = sdly_date + timedelta(days=7)
                sdly_window = working_df[
                    (working_df['date'] >= sdly_window_start) & 
                    (working_df['date'] <= sdly_window_end) &
                    (working_df['date'].dt.dayofweek == target_dow)
                ]
                sdly_exact = sdly_window['customers'].mean() if not sdly_window.empty else None
            
            # ============================================================
            # STEP 2: APPLY MEASURED GROWTH (from ACTUAL recent data)
            # ============================================================
            growth_rate = growth_rates['composite']
            
            if sdly_exact is not None and sdly_exact > 0:
                growth_adjusted_sdly = sdly_exact * (1 + growth_rate)
            else:
                # No SDLY data: use recent DOW average as base
                growth_adjusted_sdly = dow_patterns.get(target_dow, {}).get('mean', 0)
                if growth_adjusted_sdly == 0:
                    growth_adjusted_sdly = working_df.tail(14)['customers'].mean()
            
            # ============================================================
            # STEP 3: DOW FINE-TUNING (from recent 4 weeks)
            # ============================================================
            # If recent Tuesdays average 5% above overall recent mean,
            # adjust this Tuesday's forecast accordingly
            dow_info = dow_patterns.get(target_dow, {})
            dow_adjustment = dow_info.get('relative_factor', 1.0)
            
            # But don't let DOW adjustment override the growth-adjusted SDLY too much
            # The DOW factor adjusts around the growth-adjusted base
            dow_adjusted = growth_adjusted_sdly * dow_adjustment
            
            # ============================================================
            # STEP 4: ML MODEL RESIDUAL (small correction, CAPPED ±15%)
            # ============================================================
            features = compute_single_row_features(working_df, target_date, self.events_df)
            X_pred = pd.DataFrame([{col: features.get(col, 0) for col in self.feature_cols}])
            model_raw = max(0, self.model.predict(X_pred)[0])
            
            # ML correction = how different is the model's opinion from our anchor?
            if dow_adjusted > 0:
                ml_ratio = model_raw / dow_adjusted
                # Cap the ML correction at ±15% (it's a nudge, not the driver)
                ml_ratio_capped = np.clip(ml_ratio, 0.85, 1.15)
                ml_corrected = dow_adjusted * ml_ratio_capped
            else:
                ml_corrected = model_raw
            
            # ============================================================
            # STEP 5: RECENT ACTUALS SANITY CHECK
            # ============================================================
            # If DOW-specific recent actuals exist, don't deviate more than 20%
            dow_recent_mean = dow_info.get('mean', 0)
            if dow_recent_mean > 0:
                lower_bound = dow_recent_mean * 0.80
                upper_bound = dow_recent_mean * 1.20
                # Soft clamp: pull toward bounds rather than hard clip
                if ml_corrected < lower_bound:
                    ml_corrected = (ml_corrected + lower_bound) / 2
                elif ml_corrected > upper_bound:
                    ml_corrected = (ml_corrected + upper_bound) / 2
            
            # ============================================================
            # STEP 6: SELF-CORRECTION (from past forecast errors)
            # ============================================================
            final_pred = self.self_corrector.apply_correction(ml_corrected, target_date)
            
            # ============================================================
            # STEP 7: SPECIAL DATE MULTIPLIERS (events, holidays)
            # ============================================================
            final_pred = self._apply_multipliers(final_pred, target_date, features)
            
            # Ensure non-negative
            final_pred = max(0, final_pred)
            
            # ============================================================
            # CONFIDENCE INTERVALS (based on residual distribution)
            # ============================================================
            horizon_factor = 1 + (h - 1) * 0.03
            ci_margin = customer_residual_std * 1.645 * horizon_factor
            ci_lower = max(0, final_pred - ci_margin)
            ci_upper = final_pred + ci_margin
            
            # Context for diagnostics
            sdly_display = sdly_exact if sdly_exact is not None else 0
            
            predictions.append({
                'ds': target_date,
                'forecast_customers': int(round(final_pred)),
                'customers_lower': int(round(ci_lower)),
                'customers_upper': int(round(ci_upper)),
                # Component breakdown for decomposition chart
                'sdly_base': sdly_display,
                'growth_adjusted_sdly': growth_adjusted_sdly,
                'dow_adjusted': dow_adjusted,
                'model_prediction': model_raw,
                'ml_corrected': ml_corrected,
                # Legacy fields for compatibility
                'sdly_prediction': growth_adjusted_sdly,
                'recent_prediction': dow_recent_mean if dow_recent_mean > 0 else ml_corrected,
                'blended_prediction': ml_corrected,
                'corrected_prediction': final_pred,
                'yoy_growth': 1 + growth_rate,
                'recent_momentum': dow_adjustment,
                'trend_direction': 'up' if growth_rate > 0.02 else ('down' if growth_rate < -0.05 else 'neutral'),
                'trend_confidence': min(90, 40 + abs(growth_rate) * 500),
                'trend_adjustment': 1 + growth_rate
            })
            
            # Update working_df for next iteration (no NaN!)
            estimated_sales = final_pred * recent_atv_median
            working_df = pd.concat([
                working_df,
                pd.DataFrame({
                    'date': [target_date],
                    'customers': [final_pred],
                    'sales': [estimated_sales],
                    'atv': [recent_atv_median]
                })
            ], ignore_index=True)
        
        return pd.DataFrame(predictions)
    
    def _measure_yoy_growth(self, df):
        """
        PRECISELY measure how much better/worse we are vs last year.
        
        Uses multiple windows for robustness, weighted toward recent data.
        This is THE key metric — if this says +8%, our forecast should
        be ~8% above SDLY.
        """
        end_date = df['date'].max()
        rates = {}
        
        for label, days in [('7d', 7), ('14d', 14), ('30d', 30)]:
            # This year
            ty_data = df[df['date'] > end_date - timedelta(days=days)]
            # Same period last year (364 days ago for DOW alignment)
            ly_end = end_date - timedelta(days=364)
            ly_start = ly_end - timedelta(days=days)
            ly_data = df[(df['date'] > ly_start) & (df['date'] <= ly_end)]
            
            if not ty_data.empty and not ly_data.empty:
                ty_mean = ty_data['customers'].mean()
                ly_mean = ly_data['customers'].mean()
                rates[label] = (ty_mean - ly_mean) / ly_mean if ly_mean > 0 else 0
            else:
                rates[label] = 0
        
        # Composite: weighted average (recent matters more)
        # 7d × 3 + 14d × 2 + 30d × 1
        weights = {'7d': 3, '14d': 2, '30d': 1}
        total_weight = sum(weights.values())
        composite = sum(rates.get(k, 0) * w for k, w in weights.items()) / total_weight
        
        rates['composite'] = composite
        return rates
    
    def _compute_dow_patterns(self, df):
        """
        Compute day-of-week specific patterns from last 4 weeks.
        
        Returns for each DOW: mean customers, and relative factor
        (how much this DOW differs from overall recent average).
        """
        end_date = df['date'].max()
        recent_4w = df[df['date'] > end_date - timedelta(days=28)]
        
        if recent_4w.empty:
            return {}
        
        overall_mean = recent_4w['customers'].mean()
        
        patterns = {}
        for dow in range(7):
            dow_data = recent_4w[recent_4w['date'].dt.dayofweek == dow]['customers']
            if not dow_data.empty:
                dow_mean = dow_data.mean()
                patterns[dow] = {
                    'mean': dow_mean,
                    'count': len(dow_data),
                    'std': dow_data.std() if len(dow_data) > 1 else 0,
                    'relative_factor': dow_mean / overall_mean if overall_mean > 0 else 1.0
                }
            else:
                patterns[dow] = {
                    'mean': overall_mean,
                    'count': 0,
                    'std': 0,
                    'relative_factor': 1.0
                }
        
        return patterns
        """
        Estimate prediction residual standard deviation for confidence intervals.
        Uses recent data holdout to compute empirical residuals.
        """
        try:
            # Hold out last 30 days, predict, measure residuals
            if len(historical_df) < 90:
                return historical_df['customers'].std() * 0.15
            
            holdout_size = 30
            train_part = historical_df.iloc[:-holdout_size]
            test_part = historical_df.iloc[-holdout_size:]
            
            # Quick prediction using rolling means as proxy
            recent_mean = train_part.tail(14)['customers'].mean()
            dow_means = train_part.groupby(train_part['date'].dt.dayofweek)['customers'].mean()
            
            residuals = []
            for _, row in test_part.iterrows():
                dow = row['date'].weekday()
                pred = dow_means.get(dow, recent_mean)
                residuals.append(row['customers'] - pred)
            
            return np.std(residuals)
        except Exception:
            return historical_df['customers'].std() * 0.15
    
    def _apply_multipliers(self, base_pred, date, features_row):
        """Apply learned multipliers for special dates. Accepts dict or Series."""
        multiplier = 1.0
        
        # Handle both dict and Series
        def _get(key, default=0):
            if isinstance(features_row, dict):
                return features_row.get(key, default)
            return features_row.get(key, default) if hasattr(features_row, 'get') else default
        
        # New Year
        if date.month == 1 and date.day <= 2:
            multiplier *= self.multipliers.get('new_year', 1.3)
        
        # Christmas
        if date.month == 12 and date.day >= 20:
            multiplier *= self.multipliers.get('christmas', 1.15)
        
        # Event day
        if _get('is_event', 0) == 1:
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
        print("COGNITIVE FORECASTING ENGINE v12.2 (SDLY-ANCHORED)")
        print("=" * 60)
        
        # Step 0: Intelligent pre-analysis (if available)
        if self.cognitive:
            print("\n[0/3] Running Cognitive Analysis...")
            self.analysis = self.cognitive.analyze_and_prepare(historical_df, events_df)
            
            # Check for regime changes
            if self.analysis['regime'].get('regime_change'):
                print("⚠️  REGIME CHANGE DETECTED!")
                print(f"    {self.analysis['regime'].get('recommendation')}")
            
            # Check for anomalies and APPLY handling decisions
            anomalies = self.analysis['anomalies']
            n_anomalies = len(anomalies)
            if n_anomalies > 0:
                print(f"📊 Detected {n_anomalies} anomalies in historical data")
                
                # Actually apply anomaly handling to clean training data
                dates_to_exclude = []
                dates_to_cap = []
                for _, anom in anomalies.iterrows():
                    decision, reason = self.cognitive.anomaly_intel.decide_anomaly_handling(
                        anom, events_df
                    )
                    if decision == 'exclude':
                        dates_to_exclude.append(anom['date'])
                        print(f"    Excluding {anom['date'].strftime('%Y-%m-%d')}: {reason}")
                    elif decision == 'adjust':
                        dates_to_cap.append(anom['date'])
                        print(f"    Capping {anom['date'].strftime('%Y-%m-%d')}: {reason}")
                
                # Apply exclusions
                if dates_to_exclude:
                    historical_df = historical_df[~historical_df['date'].isin(dates_to_exclude)].copy()
                    print(f"    Removed {len(dates_to_exclude)} anomalous records from training")
                
                # Apply caps (cap at 3 std from rolling mean)
                if dates_to_cap:
                    rolling_mean = historical_df['customers'].rolling(30, min_periods=7).mean()
                    rolling_std = historical_df['customers'].rolling(30, min_periods=7).std()
                    upper_cap = rolling_mean + 3 * rolling_std
                    lower_cap = rolling_mean - 3 * rolling_std
                    
                    cap_mask = historical_df['date'].isin(dates_to_cap)
                    historical_df.loc[cap_mask, 'customers'] = historical_df.loc[cap_mask, 'customers'].clip(
                        lower=lower_cap[cap_mask], upper=upper_cap[cap_mask]
                    )
                    print(f"    Capped {len(dates_to_cap)} anomalous values")
            
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
        
        # Pass validation MAPE to cognitive for real confidence scoring
        if self.cognitive and hasattr(self.customer_forecaster, 'validation_mape'):
            self.cognitive._validation_mape = self.customer_forecaster.validation_mape
        
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
        
        # Calculate sales with PROPER confidence intervals
        # FIXED: Propagate uncertainty from BOTH customer and ATV forecasts
        result['forecast_sales'] = result['forecast_customers'] * result['forecast_atv']
        
        # Sales CI: combine customer CI with ATV CI
        if 'customers_lower' in result.columns:
            result['sales_lower'] = result['customers_lower'] * result['atv_lower']
            result['sales_upper'] = result['customers_upper'] * result['atv_upper']
        else:
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
