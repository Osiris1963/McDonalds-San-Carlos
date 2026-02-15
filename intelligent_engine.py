# intelligent_engine.py - COGNITIVE FORECASTING ENGINE v11.0
# This module adds "thinking" capabilities to the forecaster:
# 1. Automatic Learning Pipeline - learns when new data arrives
# 2. Anomaly Intelligence - detects and handles outliers smartly
# 3. Regime Detection - identifies when business patterns change
# 4. Multi-Scenario Forecasting - optimistic/pessimistic/realistic
# 5. Explanation Generation - explains WHY predictions were made
# 6. Confidence Scoring - knows when to trust itself

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')


class IntelligentLearningEngine:
    """
    Automatic learning engine that triggers calibration when new data arrives.
    
    Thinks like: "I see new data came in. Let me analyze what I got wrong,
    understand WHY I was wrong, and adjust my approach accordingly."
    """
    
    def __init__(self, db_client=None):
        self.db_client = db_client
        self.learning_log = []
        self.performance_history = []
        self.last_calibration = None
        self.calibration_triggers = {
            'new_data_count': 0,
            'error_threshold_breached': False,
            'pattern_shift_detected': False
        }
        
    def check_for_new_data(self, historical_df):
        """
        Detect if new data has been added since last calibration.
        Automatically triggers learning if conditions are met.
        """
        if self.db_client is None:
            return {'should_recalibrate': False, 'reason': 'No database connection'}
        
        current_max_date = historical_df['date'].max()
        
        # Load last known state
        try:
            state_doc = self.db_client.collection('ai_state').document('learning_state').get()
            if state_doc.exists:
                state = state_doc.to_dict()
                last_known_date = pd.to_datetime(state.get('last_data_date'))
                last_calibration = pd.to_datetime(state.get('last_calibration'))
                
                # Calculate new days
                new_days = (current_max_date - last_known_date).days
                days_since_calibration = (pd.to_datetime('now') - last_calibration).days
                
                should_recalibrate = (
                    new_days >= 7 or  # At least 7 new days
                    days_since_calibration >= 14 or  # Been 2 weeks
                    self._check_performance_degradation()  # Performance dropped
                )
                
                return {
                    'should_recalibrate': should_recalibrate,
                    'new_days': new_days,
                    'days_since_calibration': days_since_calibration,
                    'reason': self._get_recalibration_reason(new_days, days_since_calibration)
                }
            else:
                return {'should_recalibrate': True, 'reason': 'First time setup'}
                
        except Exception as e:
            return {'should_recalibrate': True, 'reason': f'State check failed: {e}'}
    
    def _check_performance_degradation(self):
        """Check if recent forecast accuracy has dropped significantly."""
        if len(self.performance_history) < 14:
            return False
        
        recent_7 = np.mean([p['mape'] for p in self.performance_history[-7:]])
        previous_7 = np.mean([p['mape'] for p in self.performance_history[-14:-7]])
        
        # If accuracy dropped by more than 5 percentage points
        return (recent_7 - previous_7) > 5
    
    def _get_recalibration_reason(self, new_days, days_since_cal):
        """Generate human-readable reason for recalibration."""
        reasons = []
        if new_days >= 7:
            reasons.append(f"{new_days} new days of data available")
        if days_since_cal >= 14:
            reasons.append(f"{days_since_cal} days since last calibration")
        if self._check_performance_degradation():
            reasons.append("Recent forecast accuracy has dropped")
        return "; ".join(reasons) if reasons else "Routine maintenance"
    
    def save_learning_state(self, historical_df, calibration_results):
        """Save the current learning state to Firestore."""
        if self.db_client is None:
            return False
        
        try:
            state = {
                'last_data_date': historical_df['date'].max(),
                'last_calibration': pd.to_datetime('now'),
                'calibration_results': calibration_results,
                'model_version': '11.0',
                'total_training_samples': len(historical_df)
            }
            
            self.db_client.collection('ai_state').document('learning_state').set(state)
            return True
        except Exception as e:
            print(f"Failed to save learning state: {e}")
            return False


class AnomalyIntelligence:
    """
    Smart anomaly detection that understands CONTEXT.
    
    Thinks like: "This day has unusually high sales. Is it because:
    A) Data entry error? B) Special event? C) Genuine trend shift?
    Let me investigate and decide how to handle it."
    """
    
    def __init__(self):
        self.anomaly_log = []
        self.handled_anomalies = {}
        
    def detect_anomalies(self, df, column='customers', sensitivity=2.5):
        """
        Detect anomalies using multiple methods and contextual analysis.
        """
        df = df.copy().sort_values('date')
        anomalies = []
        
        # Method 1: Statistical (Z-score with rolling window)
        rolling_mean = df[column].rolling(30, min_periods=7).mean()
        rolling_std = df[column].rolling(30, min_periods=7).std()
        z_scores = (df[column] - rolling_mean) / rolling_std
        
        # Method 2: Day-of-week adjusted
        dow_stats = df.groupby(df['date'].dt.dayofweek)[column].agg(['mean', 'std'])
        df['dow'] = df['date'].dt.dayofweek
        df['dow_z'] = df.apply(
            lambda x: (x[column] - dow_stats.loc[x['dow'], 'mean']) / dow_stats.loc[x['dow'], 'std']
            if dow_stats.loc[x['dow'], 'std'] > 0 else 0,
            axis=1
        )
        
        # Method 3: Sequential anomaly (sudden jumps)
        df['pct_change'] = df[column].pct_change().abs()
        
        for idx, row in df.iterrows():
            is_anomaly = False
            anomaly_type = None
            confidence = 0
            
            # Check Z-score
            if abs(z_scores.loc[idx]) > sensitivity:
                is_anomaly = True
                anomaly_type = 'statistical_outlier'
                confidence = min(abs(z_scores.loc[idx]) / 5, 1.0)
            
            # Check day-of-week adjusted
            elif abs(row['dow_z']) > sensitivity + 0.5:
                is_anomaly = True
                anomaly_type = 'dow_anomaly'
                confidence = min(abs(row['dow_z']) / 5, 1.0)
            
            # Check sequential jump
            elif row['pct_change'] > 0.5:  # >50% change
                is_anomaly = True
                anomaly_type = 'sudden_jump'
                confidence = min(row['pct_change'], 1.0)
            
            if is_anomaly:
                anomalies.append({
                    'date': row['date'],
                    'value': row[column],
                    'expected': rolling_mean.loc[idx],
                    'z_score': z_scores.loc[idx],
                    'type': anomaly_type,
                    'confidence': confidence,
                    'explanation': self._generate_anomaly_explanation(row, anomaly_type, z_scores.loc[idx])
                })
        
        return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()
    
    def _generate_anomaly_explanation(self, row, anomaly_type, z_score):
        """Generate human-readable explanation for the anomaly."""
        date = row['date']
        value = row['customers'] if 'customers' in row else row.get('sales', 0)
        
        # Check for known reasons
        explanations = []
        
        # Holiday check
        if date.month == 1 and date.day <= 2:
            explanations.append("New Year period - high traffic expected")
        elif date.month == 12 and date.day >= 20:
            explanations.append("Christmas season - high traffic expected")
        elif date.day in [14, 15, 16, 29, 30, 31, 1, 2]:
            explanations.append("Payday period - elevated traffic normal")
        
        # Day of week
        if date.weekday() >= 5:
            explanations.append("Weekend - typically different pattern")
        
        # Z-score interpretation
        if z_score > 3:
            explanations.append(f"Extremely high ({z_score:.1f} std above normal)")
        elif z_score > 2:
            explanations.append(f"Unusually high ({z_score:.1f} std above normal)")
        elif z_score < -3:
            explanations.append(f"Extremely low ({abs(z_score):.1f} std below normal)")
        elif z_score < -2:
            explanations.append(f"Unusually low ({abs(z_score):.1f} std below normal)")
        
        if not explanations:
            explanations.append("Unexpected variation - investigate cause")
        
        return "; ".join(explanations)
    
    def decide_anomaly_handling(self, anomaly_row, events_df=None):
        """
        Intelligently decide how to handle each anomaly.
        
        Returns: 'include', 'exclude', 'adjust', or 'flag'
        """
        date = anomaly_row['date']
        anomaly_type = anomaly_row['type']
        z_score = anomaly_row['z_score']
        
        # Check if it's a known event
        is_known_event = False
        if events_df is not None and not events_df.empty:
            ev_dates = pd.to_datetime(events_df['date']).dt.normalize()
            is_known_event = date in ev_dates.values
        
        # Decision logic
        if is_known_event:
            # Known event - include but note it
            return 'include', "Known event - keeping in training data"
        
        if anomaly_type == 'sudden_jump' and abs(z_score) > 4:
            # Extreme jump with no explanation - likely data error
            return 'exclude', "Extreme unexplained jump - possible data error"
        
        if date.month == 1 and date.day <= 2:
            # New Year - always include
            return 'include', "New Year pattern - important seasonal data"
        
        if abs(z_score) > 4:
            # Very extreme with no explanation
            return 'adjust', f"Extreme outlier - capping at 3 std from mean"
        
        # Default: include but flag
        return 'flag', "Unusual but keeping - will monitor impact"


class RegimeDetector:
    """
    Detects when business patterns fundamentally change.
    
    Thinks like: "The relationship between weekday and sales seems different
    now than it was 3 months ago. Has something fundamentally changed?
    Should I weight recent data even more heavily?"
    """
    
    def __init__(self):
        self.regimes = []
        self.current_regime = None
        
    def detect_regime_change(self, df, lookback_days=90, min_regime_length=30):
        """
        Detect if the business is operating in a different "regime"
        (e.g., post-renovation, new competitor, economic shift).
        """
        if len(df) < lookback_days:
            return {'regime_change': False, 'confidence': 0}
        
        df = df.copy().sort_values('date')
        recent = df.tail(min_regime_length)
        historical = df.iloc[-(lookback_days):-(min_regime_length)]
        
        changes_detected = []
        
        # Check 1: Mean shift
        recent_mean = recent['customers'].mean()
        hist_mean = historical['customers'].mean()
        mean_change = (recent_mean - hist_mean) / hist_mean
        
        if abs(mean_change) > 0.15:  # >15% change in average
            changes_detected.append({
                'type': 'mean_shift',
                'magnitude': mean_change,
                'description': f"Average traffic {'increased' if mean_change > 0 else 'decreased'} by {abs(mean_change)*100:.1f}%"
            })
        
        # Check 2: Volatility shift
        recent_cv = recent['customers'].std() / recent['customers'].mean()
        hist_cv = historical['customers'].std() / historical['customers'].mean()
        vol_change = (recent_cv - hist_cv) / hist_cv
        
        if abs(vol_change) > 0.30:  # >30% change in volatility
            changes_detected.append({
                'type': 'volatility_shift',
                'magnitude': vol_change,
                'description': f"Traffic volatility {'increased' if vol_change > 0 else 'decreased'} by {abs(vol_change)*100:.1f}%"
            })
        
        # Check 3: Day-of-week pattern shift
        recent_dow = recent.groupby(recent['date'].dt.dayofweek)['customers'].mean()
        hist_dow = historical.groupby(historical['date'].dt.dayofweek)['customers'].mean()
        dow_correlation = recent_dow.corr(hist_dow)
        
        if dow_correlation < 0.85:  # Weak correlation = pattern changed
            changes_detected.append({
                'type': 'pattern_shift',
                'magnitude': 1 - dow_correlation,
                'description': f"Day-of-week patterns have changed (correlation: {dow_correlation:.2f})"
            })
        
        # Check 4: Trend direction change
        hist_trend = np.polyfit(range(len(historical)), historical['customers'].values, 1)[0]
        recent_trend = np.polyfit(range(len(recent)), recent['customers'].values, 1)[0]
        
        if (hist_trend > 0 and recent_trend < 0) or (hist_trend < 0 and recent_trend > 0):
            changes_detected.append({
                'type': 'trend_reversal',
                'magnitude': abs(recent_trend - hist_trend),
                'description': f"Trend reversed from {'growing' if hist_trend > 0 else 'declining'} to {'growing' if recent_trend > 0 else 'declining'}"
            })
        
        regime_change = len(changes_detected) >= 2 or any(c['type'] == 'trend_reversal' for c in changes_detected)
        
        return {
            'regime_change': regime_change,
            'confidence': min(len(changes_detected) / 3, 1.0),
            'changes': changes_detected,
            'recommendation': self._get_regime_recommendation(changes_detected)
        }
    
    def _get_regime_recommendation(self, changes):
        """Generate actionable recommendation based on detected changes."""
        if not changes:
            return "No significant regime change detected. Continue with current model."
        
        recommendations = []
        
        for change in changes:
            if change['type'] == 'mean_shift':
                if change['magnitude'] > 0:
                    recommendations.append("Consider increasing baseline forecasts")
                else:
                    recommendations.append("Consider decreasing baseline forecasts")
            
            elif change['type'] == 'volatility_shift':
                if change['magnitude'] > 0:
                    recommendations.append("Widen confidence intervals for forecasts")
                else:
                    recommendations.append("Forecasts should be more precise now")
            
            elif change['type'] == 'pattern_shift':
                recommendations.append("Increase weight on recent data (last 4 weeks)")
            
            elif change['type'] == 'trend_reversal':
                recommendations.append("IMPORTANT: Business trend has reversed - prioritize recent data heavily")
        
        return " | ".join(recommendations)


class MultiScenarioForecaster:
    """
    Generates multiple forecast scenarios instead of single point estimates.
    
    Thinks like: "The most likely outcome is X, but if the recent trend
    continues strongly it could be Y, and if there's a slowdown it could be Z.
    Here are all three scenarios with their probabilities."
    """
    
    def __init__(self):
        self.scenarios = {}
        
    def generate_scenarios(self, base_forecast, recent_momentum, volatility, regime_info):
        """
        Generate optimistic, realistic, and pessimistic scenarios.
        """
        scenarios = []
        
        # Scenario 1: Realistic (base forecast)
        scenarios.append({
            'name': 'Realistic',
            'probability': 0.50,
            'forecast': base_forecast,
            'description': 'Most likely outcome based on all available data',
            'color': '#3B82F6'
        })
        
        # Scenario 2: Optimistic
        # If momentum is positive, amplify it; otherwise, assume improvement
        if recent_momentum > 1:
            optimistic_multiplier = 1 + (recent_momentum - 1) * 1.5
        else:
            optimistic_multiplier = 1 + volatility * 0.5
        
        scenarios.append({
            'name': 'Optimistic',
            'probability': 0.25,
            'forecast': int(base_forecast * optimistic_multiplier),
            'description': 'If positive trends continue or conditions improve',
            'color': '#10B981'
        })
        
        # Scenario 3: Pessimistic
        if recent_momentum < 1:
            pessimistic_multiplier = 1 - (1 - recent_momentum) * 1.5
        else:
            pessimistic_multiplier = 1 - volatility * 0.5
        
        pessimistic_multiplier = max(0.5, pessimistic_multiplier)  # Floor at 50%
        
        scenarios.append({
            'name': 'Pessimistic',
            'probability': 0.25,
            'forecast': int(base_forecast * pessimistic_multiplier),
            'description': 'If negative trends emerge or conditions worsen',
            'color': '#EF4444'
        })
        
        # Adjust probabilities based on regime info
        if regime_info.get('regime_change', False):
            # More uncertainty = more weight to extreme scenarios
            scenarios[0]['probability'] = 0.40  # Realistic
            scenarios[1]['probability'] = 0.30  # Optimistic
            scenarios[2]['probability'] = 0.30  # Pessimistic
        
        return scenarios
    
    def calculate_expected_value(self, scenarios):
        """Calculate probability-weighted expected value."""
        return sum(s['forecast'] * s['probability'] for s in scenarios)


class ExplanationGenerator:
    """
    Generates human-readable explanations for forecasts.
    
    Thinks like: "Let me explain this forecast in plain English so the
    business owner understands WHY I'm predicting this number and what
    factors are most important."
    """
    
    def __init__(self):
        self.explanation_templates = {}
        
    def generate_forecast_explanation(self, forecast_row, feature_importance, 
                                       contextual_features, recent_features, 
                                       self_correction_factors):
        """
        Generate a comprehensive explanation for a single forecast.
        """
        date = forecast_row['ds']
        prediction = forecast_row['forecast_customers']
        
        explanations = []
        
        # 1. Date context
        day_name = date.strftime('%A')
        date_str = date.strftime('%B %d, %Y')
        explanations.append(f"**Forecasting {day_name}, {date_str}**")
        
        # 2. Component breakdown
        model_pred = forecast_row.get('model_prediction', prediction)
        sdly_pred = forecast_row.get('sdly_prediction', prediction)
        recent_pred = forecast_row.get('recent_prediction', prediction)
        
        explanations.append(f"\n**Prediction Components:**")
        explanations.append(f"- ML Model suggests: {model_pred:,.0f} customers")
        explanations.append(f"- Same period last year (adjusted): {sdly_pred:,.0f} customers")
        explanations.append(f"- Recent 8-week trend suggests: {recent_pred:,.0f} customers")
        explanations.append(f"- **Blended forecast: {prediction:,} customers**")
        
        # 3. Key factors
        explanations.append(f"\n**Key Factors Considered:**")
        
        # Day of week
        if date.weekday() >= 5:
            explanations.append(f"- Weekend day (typically {'higher' if date.weekday() == 5 else 'moderate'} traffic)")
        else:
            explanations.append(f"- Weekday ({day_name})")
        
        # Payday
        if date.day in [14, 15, 16]:
            explanations.append("- Mid-month payday period (typically +5-10% traffic)")
        elif date.day in [29, 30, 31, 1, 2]:
            explanations.append("- End-month payday period (typically +5-10% traffic)")
        
        # Special dates
        if date.month == 1 and date.day <= 2:
            explanations.append("- New Year period (historically high traffic)")
        elif date.month == 12 and date.day >= 20:
            explanations.append("- Christmas season (elevated traffic expected)")
        
        # YoY growth
        yoy = forecast_row.get('yoy_growth', 1.0)
        if yoy > 1.05:
            explanations.append(f"- Year-over-year growth trend: +{(yoy-1)*100:.1f}%")
        elif yoy < 0.95:
            explanations.append(f"- Year-over-year decline: {(yoy-1)*100:.1f}%")
        
        # Recent momentum
        momentum = forecast_row.get('recent_momentum', 1.0)
        if momentum > 1.05:
            explanations.append(f"- Recent momentum is positive: +{(momentum-1)*100:.1f}%")
        elif momentum < 0.95:
            explanations.append(f"- Recent momentum is negative: {(momentum-1)*100:.1f}%")
        
        # 4. Self-correction applied
        if self_correction_factors:
            explanations.append(f"\n**Self-Correction Applied:**")
            dow_corr = self_correction_factors.get('dow', {}).get(date.weekday(), 1.0)
            if abs(dow_corr - 1.0) > 0.01:
                explanations.append(f"- {day_name} adjustment: {(dow_corr-1)*100:+.1f}% (learned from past errors)")
        
        # 5. Confidence statement
        explanations.append(f"\n**Confidence:** ")
        if abs(model_pred - sdly_pred) < prediction * 0.1 and abs(model_pred - recent_pred) < prediction * 0.1:
            explanations.append("HIGH - All prediction methods agree closely")
        elif abs(model_pred - sdly_pred) < prediction * 0.2:
            explanations.append("MEDIUM - Methods show some variation")
        else:
            explanations.append("LOWER - Methods show significant variation; wider outcome range possible")
        
        return "\n".join(explanations)
    
    def generate_daily_briefing(self, forecast_df, anomalies_df, regime_info):
        """
        Generate an executive daily briefing.
        """
        total_customers = forecast_df['forecast_customers'].sum()
        total_sales = forecast_df['forecast_sales'].sum()
        peak_day = forecast_df.loc[forecast_df['forecast_sales'].idxmax()]
        
        briefing = []
        briefing.append("# 📊 AI Forecast Daily Briefing\n")
        
        # Summary
        briefing.append("## 15-Day Outlook Summary")
        briefing.append(f"- **Total Projected Customers:** {total_customers:,}")
        briefing.append(f"- **Total Projected Sales:** ₱{total_sales:,.0f}")
        briefing.append(f"- **Peak Day:** {peak_day['ds'].strftime('%A, %B %d')} (₱{peak_day['forecast_sales']:,.0f})")
        
        # Trend insight
        first_week = forecast_df.head(7)['forecast_sales'].mean()
        second_week = forecast_df.tail(8)['forecast_sales'].mean()
        week_trend = (second_week - first_week) / first_week * 100
        
        briefing.append(f"\n## Trend Analysis")
        if week_trend > 5:
            briefing.append(f"📈 **GROWING:** Second week projected {week_trend:.1f}% higher than first week")
        elif week_trend < -5:
            briefing.append(f"📉 **DECLINING:** Second week projected {abs(week_trend):.1f}% lower than first week")
        else:
            briefing.append(f"➡️ **STABLE:** Relatively flat across the 15-day period")
        
        # Regime warning
        if regime_info.get('regime_change', False):
            briefing.append(f"\n## ⚠️ Business Pattern Change Detected")
            for change in regime_info.get('changes', []):
                briefing.append(f"- {change['description']}")
            briefing.append(f"\n**Recommendation:** {regime_info.get('recommendation', 'Monitor closely')}")
        
        # Anomaly alerts
        if anomalies_df is not None and not anomalies_df.empty:
            briefing.append(f"\n## 🔔 Data Anomalies Detected ({len(anomalies_df)})")
            for _, anom in anomalies_df.head(3).iterrows():
                briefing.append(f"- {anom['date'].strftime('%Y-%m-%d')}: {anom['explanation']}")
        
        return "\n".join(briefing)


class ConfidenceScorer:
    """
    Calculates how confident the model should be in its predictions.
    
    Thinks like: "I'm very confident about this Tuesday forecast because
    my recent Tuesday predictions have been accurate. But I'm less
    confident about this holiday because I've never seen this exact
    situation before."
    """
    
    def __init__(self):
        self.confidence_factors = {}
        
    def calculate_confidence(self, forecast_row, historical_accuracy, 
                            data_quality_score, component_agreement):
        """
        Calculate a confidence score (0-100) for a forecast.
        """
        factors = {}
        
        # Factor 1: Historical accuracy for this day type (0-30 points)
        dow = forecast_row['ds'].weekday()
        dow_accuracy = historical_accuracy.get(f'dow_{dow}', 85)
        factors['historical_accuracy'] = min(30, dow_accuracy * 0.35)
        
        # Factor 2: Component agreement (0-30 points)
        # If model, SDLY, and recent all agree, high confidence
        model = forecast_row.get('model_prediction', 0)
        sdly = forecast_row.get('sdly_prediction', 0)
        recent = forecast_row.get('recent_prediction', 0)
        
        if model > 0:
            max_diff = max(abs(model - sdly), abs(model - recent), abs(sdly - recent))
            agreement_ratio = 1 - (max_diff / model)
            factors['component_agreement'] = max(0, min(30, agreement_ratio * 35))
        else:
            factors['component_agreement'] = 15
        
        # Factor 3: Data quality (0-20 points)
        factors['data_quality'] = min(20, data_quality_score * 0.2)
        
        # Factor 4: Forecast horizon (0-20 points)
        # Closer = more confident
        horizon = (forecast_row['ds'] - pd.to_datetime('today')).days
        horizon_confidence = max(0, 20 - horizon * 1.5)
        factors['horizon'] = horizon_confidence
        
        total_confidence = sum(factors.values())
        
        return {
            'score': round(total_confidence),
            'grade': self._confidence_grade(total_confidence),
            'factors': factors,
            'interpretation': self._interpret_confidence(total_confidence, factors)
        }
    
    def _confidence_grade(self, score):
        if score >= 85:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 55:
            return 'C'
        elif score >= 40:
            return 'D'
        else:
            return 'F'
    
    def _interpret_confidence(self, score, factors):
        if score >= 85:
            return "Very high confidence - historical accuracy is strong and all prediction methods agree"
        elif score >= 70:
            return "Good confidence - reasonable basis for planning"
        elif score >= 55:
            return "Moderate confidence - consider wider planning range"
        elif score >= 40:
            return "Lower confidence - significant uncertainty exists"
        else:
            return "Low confidence - use with caution, consider multiple scenarios"


class CognitiveForecaster:
    """
    Master class that combines all intelligent components.
    This is the "brain" that orchestrates thinking about forecasts.
    """
    
    def __init__(self, db_client=None):
        self.db_client = db_client
        self.learning_engine = IntelligentLearningEngine(db_client)
        self.anomaly_intel = AnomalyIntelligence()
        self.regime_detector = RegimeDetector()
        self.scenario_gen = MultiScenarioForecaster()
        self.explainer = ExplanationGenerator()
        self.confidence_scorer = ConfidenceScorer()
        
        self.insights = []
        self.warnings = []
        
    def analyze_and_prepare(self, historical_df, events_df=None):
        """
        Comprehensive analysis before forecasting.
        Returns insights and recommendations.
        """
        analysis = {
            'data_quality': self._assess_data_quality(historical_df),
            'anomalies': self.anomaly_intel.detect_anomalies(historical_df),
            'regime': self.regime_detector.detect_regime_change(historical_df),
            'learning_status': self.learning_engine.check_for_new_data(historical_df),
            'recommendations': []
        }
        
        # Generate recommendations
        if analysis['learning_status'].get('should_recalibrate'):
            analysis['recommendations'].append({
                'priority': 'HIGH',
                'action': 'Recalibrate model',
                'reason': analysis['learning_status'].get('reason')
            })
        
        if analysis['regime'].get('regime_change'):
            analysis['recommendations'].append({
                'priority': 'HIGH',
                'action': 'Review business patterns',
                'reason': analysis['regime'].get('recommendation')
            })
        
        if len(analysis['anomalies']) > 5:
            analysis['recommendations'].append({
                'priority': 'MEDIUM',
                'action': 'Review data quality',
                'reason': f"{len(analysis['anomalies'])} anomalies detected in historical data"
            })
        
        return analysis
    
    def _assess_data_quality(self, df):
        """Assess overall data quality."""
        score = 100
        issues = []
        
        # Check for missing dates
        date_range = pd.date_range(df['date'].min(), df['date'].max())
        missing_dates = len(date_range) - len(df)
        if missing_dates > 0:
            score -= min(20, missing_dates * 2)
            issues.append(f"{missing_dates} missing dates")
        
        # Check for zeros
        zero_days = (df['customers'] == 0).sum()
        if zero_days > 0:
            score -= min(15, zero_days * 3)
            issues.append(f"{zero_days} days with zero customers")
        
        # Check for negative values
        negative = (df['customers'] < 0).sum()
        if negative > 0:
            score -= 20
            issues.append(f"{negative} days with negative values (data error)")
        
        # Check recency
        days_since_last = (pd.to_datetime('today') - df['date'].max()).days
        if days_since_last > 7:
            score -= min(15, days_since_last - 7)
            issues.append(f"Data is {days_since_last} days old")
        
        return {
            'score': max(0, score),
            'grade': 'Good' if score >= 80 else 'Fair' if score >= 60 else 'Poor',
            'issues': issues
        }
    
    def generate_intelligent_forecast(self, base_forecast_df, historical_df, 
                                       self_correction_factors=None, feature_importance=None):
        """
        Enhance base forecast with intelligent features.
        """
        enhanced_forecasts = []
        
        regime_info = self.regime_detector.detect_regime_change(historical_df)
        
        for _, row in base_forecast_df.iterrows():
            # Get recent features for this date
            recent_momentum = row.get('recent_momentum', 1.0)
            volatility = historical_df['customers'].std() / historical_df['customers'].mean()
            
            # Generate scenarios
            scenarios = self.scenario_gen.generate_scenarios(
                row['forecast_customers'],
                recent_momentum,
                volatility,
                regime_info
            )
            
            # Calculate confidence using REAL validation accuracy if available
            historical_accuracy_real = {}
            if hasattr(self, '_backtest_accuracy') and self._backtest_accuracy:
                historical_accuracy_real = self._backtest_accuracy
            else:
                # Estimate from validation MAPE if available
                val_mape = getattr(self, '_validation_mape', 12)
                base_acc = 100 - val_mape
                for dow in range(7):
                    historical_accuracy_real[f'dow_{dow}'] = base_acc
            
            confidence = self.confidence_scorer.calculate_confidence(
                row,
                historical_accuracy=historical_accuracy_real,
                data_quality_score=self._assess_data_quality(historical_df).get('score', 85) if hasattr(self, '_assess_data_quality') else 85,
                component_agreement=0.9
            )
            
            # Generate explanation
            explanation = self.explainer.generate_forecast_explanation(
                row, feature_importance, {}, {}, self_correction_factors or {}
            )
            
            enhanced_forecasts.append({
                **row.to_dict(),
                'scenarios': scenarios,
                'confidence': confidence,
                'explanation': explanation,
                'expected_value': self.scenario_gen.calculate_expected_value(scenarios)
            })
        
        return enhanced_forecasts, regime_info
    
    def save_insights_to_firestore(self, insights, forecast_date):
        """Save AI insights to Firestore for learning."""
        if self.db_client is None:
            return False
        
        try:
            doc_id = forecast_date.strftime('%Y-%m-%d')
            self.db_client.collection('ai_insights').document(doc_id).set({
                'generated_at': pd.to_datetime('now'),
                'insights': insights,
                'model_version': '11.0'
            })
            return True
        except:
            return False
