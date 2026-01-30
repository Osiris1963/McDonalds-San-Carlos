# auto_learning.py - AUTOMATIC LEARNING SYSTEM v11.0
# This module handles:
# 1. Automatic detection of new data
# 2. Continuous accuracy monitoring
# 3. Automatic model recalibration
# 4. Learning from mistakes in real-time
# 5. Adaptive weight adjustment

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


class AutoLearningSystem:
    """
    Automatic learning system that continuously improves the model.
    
    Think of this as the model's "subconscious" - always running in the
    background, noticing patterns, and making adjustments.
    """
    
    def __init__(self, db_client):
        self.db_client = db_client
        self.learning_config = {
            'min_samples_for_learning': 7,
            'recalibration_threshold_mape': 15,  # Recalibrate if MAPE > 15%
            'max_correction_factor': 1.25,  # Max 25% adjustment
            'min_correction_factor': 0.75,  # Min -25% adjustment
            'learning_rate': 0.3,  # How fast to adapt (0-1)
            'momentum_decay': 0.9  # Weight decay for older errors
        }
        
    def run_learning_cycle(self, historical_df, forecast_log_df):
        """
        Main learning cycle - called whenever new actual data comes in.
        
        Steps:
        1. Compare recent forecasts to actuals
        2. Identify systematic errors
        3. Update correction factors
        4. Adjust blend weights if needed
        5. Save learned parameters
        """
        if forecast_log_df.empty or len(forecast_log_df) < self.learning_config['min_samples_for_learning']:
            return {'status': 'insufficient_data', 'samples': len(forecast_log_df)}
        
        # Step 1: Calculate errors
        errors_df = self._calculate_errors(historical_df, forecast_log_df)
        
        if errors_df.empty:
            return {'status': 'no_matchable_data'}
        
        # Step 2: Analyze error patterns
        error_patterns = self._analyze_error_patterns(errors_df)
        
        # Step 3: Generate correction factors
        corrections = self._generate_corrections(error_patterns)
        
        # Step 4: Evaluate if blend weights need adjustment
        weight_adjustments = self._evaluate_blend_weights(errors_df)
        
        # Step 5: Save to Firestore
        self._save_learning_results(corrections, weight_adjustments, error_patterns)
        
        return {
            'status': 'success',
            'samples_analyzed': len(errors_df),
            'corrections': corrections,
            'weight_adjustments': weight_adjustments,
            'overall_mape': error_patterns['overall']['mape'],
            'recommendation': self._generate_recommendation(error_patterns)
        }
    
    def _calculate_errors(self, historical_df, forecast_log_df):
        """Calculate forecast errors by matching predictions to actuals."""
        errors = []
        
        for _, forecast in forecast_log_df.iterrows():
            forecast_date = pd.to_datetime(forecast.get('forecast_for_date', forecast.get('date')))
            
            actual = historical_df[historical_df['date'] == forecast_date]
            
            if len(actual) > 0:
                actual = actual.iloc[0]
                
                pred_cust = forecast.get('predicted_customers', 0)
                actual_cust = actual.get('customers', 0)
                
                if actual_cust > 0 and pred_cust > 0:
                    errors.append({
                        'date': forecast_date,
                        'predicted': pred_cust,
                        'actual': actual_cust,
                        'error': actual_cust - pred_cust,
                        'pct_error': (actual_cust - pred_cust) / actual_cust * 100,
                        'abs_pct_error': abs(actual_cust - pred_cust) / actual_cust * 100,
                        'dayofweek': forecast_date.dayofweek(),
                        'month': forecast_date.month,
                        'day': forecast_date.day,
                        'is_payday': forecast_date.day in [14, 15, 16, 29, 30, 31, 1, 2],
                        'is_weekend': forecast_date.dayofweek() >= 5,
                        # Component predictions if available
                        'model_pred': forecast.get('model_prediction'),
                        'sdly_pred': forecast.get('sdly_prediction'),
                        'recent_pred': forecast.get('recent_prediction')
                    })
        
        return pd.DataFrame(errors)
    
    def _analyze_error_patterns(self, errors_df):
        """Identify systematic patterns in errors."""
        patterns = {
            'overall': {
                'mape': errors_df['abs_pct_error'].mean(),
                'bias': errors_df['pct_error'].mean(),  # Positive = under-predicting
                'std': errors_df['pct_error'].std(),
                'n_samples': len(errors_df)
            },
            'by_dow': {},
            'by_month': {},
            'by_payday': {},
            'by_component': {},
            'trend': {}
        }
        
        # Day of week patterns
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for dow in range(7):
            dow_errors = errors_df[errors_df['dayofweek'] == dow]
            if len(dow_errors) >= 2:
                patterns['by_dow'][dow] = {
                    'name': dow_names[dow],
                    'mape': dow_errors['abs_pct_error'].mean(),
                    'bias': dow_errors['pct_error'].mean(),
                    'n': len(dow_errors)
                }
        
        # Month patterns
        for month in errors_df['month'].unique():
            month_errors = errors_df[errors_df['month'] == month]
            if len(month_errors) >= 2:
                patterns['by_month'][month] = {
                    'mape': month_errors['abs_pct_error'].mean(),
                    'bias': month_errors['pct_error'].mean(),
                    'n': len(month_errors)
                }
        
        # Payday patterns
        payday_errors = errors_df[errors_df['is_payday']]
        non_payday_errors = errors_df[~errors_df['is_payday']]
        
        if len(payday_errors) >= 2:
            patterns['by_payday']['payday'] = {
                'mape': payday_errors['abs_pct_error'].mean(),
                'bias': payday_errors['pct_error'].mean(),
                'n': len(payday_errors)
            }
        
        if len(non_payday_errors) >= 2:
            patterns['by_payday']['non_payday'] = {
                'mape': non_payday_errors['abs_pct_error'].mean(),
                'bias': non_payday_errors['pct_error'].mean(),
                'n': len(non_payday_errors)
            }
        
        # Component performance (which prediction method is best?)
        for component in ['model_pred', 'sdly_pred', 'recent_pred']:
            valid = errors_df[errors_df[component].notna()]
            if len(valid) >= 2:
                comp_errors = abs(valid['actual'] - valid[component]) / valid['actual'] * 100
                patterns['by_component'][component] = {
                    'mape': comp_errors.mean(),
                    'n': len(valid)
                }
        
        # Error trend (getting better or worse?)
        if len(errors_df) >= 14:
            errors_df = errors_df.sort_values('date')
            recent = errors_df.tail(7)['abs_pct_error'].mean()
            older = errors_df.head(7)['abs_pct_error'].mean()
            patterns['trend'] = {
                'recent_mape': recent,
                'older_mape': older,
                'improving': recent < older,
                'change': recent - older
            }
        
        return patterns
    
    def _generate_corrections(self, patterns):
        """Generate correction factors based on patterns."""
        corrections = {
            'dow': {},
            'month': {},
            'payday': 1.0,
            'overall': 1.0
        }
        
        learning_rate = self.learning_config['learning_rate']
        max_corr = self.learning_config['max_correction_factor']
        min_corr = self.learning_config['min_correction_factor']
        
        # Day of week corrections
        for dow, data in patterns['by_dow'].items():
            bias = data['bias']
            # If bias is positive (under-predicting), we need to increase future predictions
            # If bias is negative (over-predicting), we need to decrease
            raw_correction = 1 + (bias / 100) * learning_rate
            corrections['dow'][dow] = np.clip(raw_correction, min_corr, max_corr)
        
        # Month corrections
        for month, data in patterns['by_month'].items():
            bias = data['bias']
            raw_correction = 1 + (bias / 100) * learning_rate * 0.5  # Less aggressive for months
            corrections['month'][month] = np.clip(raw_correction, min_corr, max_corr)
        
        # Payday correction
        if 'payday' in patterns['by_payday']:
            bias = patterns['by_payday']['payday']['bias']
            raw_correction = 1 + (bias / 100) * learning_rate
            corrections['payday'] = np.clip(raw_correction, min_corr, max_corr)
        
        # Overall correction
        overall_bias = patterns['overall']['bias']
        raw_correction = 1 + (overall_bias / 100) * learning_rate * 0.3  # Very conservative
        corrections['overall'] = np.clip(raw_correction, 0.95, 1.05)
        
        return corrections
    
    def _evaluate_blend_weights(self, errors_df):
        """Evaluate if blend weights should be adjusted based on component performance."""
        current_weights = {
            'model': 0.50,
            'sdly_contextual': 0.25,
            'recent_trend': 0.25
        }
        
        # Calculate MAPE for each component
        component_mapes = {}
        
        for comp_name, col_name in [('model', 'model_pred'), ('sdly', 'sdly_pred'), ('recent', 'recent_pred')]:
            valid = errors_df[errors_df[col_name].notna()]
            if len(valid) >= 5:
                mape = (abs(valid['actual'] - valid[col_name]) / valid['actual'] * 100).mean()
                component_mapes[comp_name] = mape
        
        if len(component_mapes) < 3:
            return {'adjusted': False, 'reason': 'Insufficient component data'}
        
        # Check if any component is significantly better
        best = min(component_mapes, key=component_mapes.get)
        worst = max(component_mapes, key=component_mapes.get)
        
        best_mape = component_mapes[best]
        worst_mape = component_mapes[worst]
        
        # If difference is significant (>5 percentage points), adjust weights
        if worst_mape - best_mape > 5:
            new_weights = current_weights.copy()
            
            # Increase best component's weight
            if best == 'model':
                new_weights['model'] = min(0.65, current_weights['model'] + 0.05)
                new_weights['sdly_contextual'] = (1 - new_weights['model']) / 2
                new_weights['recent_trend'] = (1 - new_weights['model']) / 2
            elif best == 'sdly':
                new_weights['sdly_contextual'] = min(0.40, current_weights['sdly_contextual'] + 0.05)
                new_weights['model'] = 0.50
                new_weights['recent_trend'] = 1 - 0.50 - new_weights['sdly_contextual']
            else:  # recent
                new_weights['recent_trend'] = min(0.40, current_weights['recent_trend'] + 0.05)
                new_weights['model'] = 0.50
                new_weights['sdly_contextual'] = 1 - 0.50 - new_weights['recent_trend']
            
            return {
                'adjusted': True,
                'old_weights': current_weights,
                'new_weights': new_weights,
                'reason': f'{best} component is performing best (MAPE: {best_mape:.1f}%)',
                'component_mapes': component_mapes
            }
        
        return {
            'adjusted': False,
            'reason': 'Component performance is balanced',
            'component_mapes': component_mapes
        }
    
    def _save_learning_results(self, corrections, weight_adjustments, patterns):
        """Save learning results to Firestore."""
        if self.db_client is None:
            return False
        
        try:
            # Save corrections
            self.db_client.collection('ai_learning').document('corrections').set({
                'updated_at': datetime.now(),
                'corrections': corrections,
                'version': '11.0'
            }, merge=True)
            
            # Save weight adjustments if needed
            if weight_adjustments.get('adjusted'):
                self.db_client.collection('ai_learning').document('blend_weights').set({
                    'updated_at': datetime.now(),
                    'weights': weight_adjustments['new_weights'],
                    'reason': weight_adjustments['reason']
                }, merge=True)
            
            # Save patterns for analysis
            self.db_client.collection('ai_learning').document('error_patterns').set({
                'updated_at': datetime.now(),
                'patterns': {
                    'overall_mape': patterns['overall']['mape'],
                    'overall_bias': patterns['overall']['bias'],
                    'trend': patterns.get('trend', {})
                }
            }, merge=True)
            
            return True
            
        except Exception as e:
            print(f"Failed to save learning results: {e}")
            return False
    
    def _generate_recommendation(self, patterns):
        """Generate human-readable recommendation."""
        mape = patterns['overall']['mape']
        bias = patterns['overall']['bias']
        trend = patterns.get('trend', {})
        
        recommendations = []
        
        # Overall accuracy assessment
        if mape <= 8:
            recommendations.append("✅ Excellent accuracy - model is performing very well")
        elif mape <= 12:
            recommendations.append("👍 Good accuracy - minor improvements possible")
        elif mape <= 15:
            recommendations.append("⚠️ Moderate accuracy - consider model tuning")
        else:
            recommendations.append("🔴 Accuracy needs improvement - recommend investigation")
        
        # Bias assessment
        if abs(bias) > 5:
            if bias > 0:
                recommendations.append(f"📊 Systematic under-prediction by {bias:.1f}% - corrections being applied")
            else:
                recommendations.append(f"📊 Systematic over-prediction by {abs(bias):.1f}% - corrections being applied")
        
        # Trend assessment
        if trend.get('improving'):
            recommendations.append(f"📈 Accuracy is improving (from {trend['older_mape']:.1f}% to {trend['recent_mape']:.1f}% MAPE)")
        elif trend.get('change', 0) > 2:
            recommendations.append(f"📉 Accuracy declining - recommend urgent review")
        
        return " | ".join(recommendations)
    
    def load_learned_corrections(self):
        """Load previously learned corrections from Firestore."""
        if self.db_client is None:
            return None
        
        try:
            doc = self.db_client.collection('ai_learning').document('corrections').get()
            if doc.exists:
                return doc.to_dict().get('corrections', {})
            return None
        except:
            return None
    
    def load_blend_weights(self):
        """Load learned blend weights from Firestore."""
        if self.db_client is None:
            return None
        
        try:
            doc = self.db_client.collection('ai_learning').document('blend_weights').get()
            if doc.exists:
                return doc.to_dict().get('weights', None)
            return None
        except:
            return None


class RealTimeLearner:
    """
    Real-time learner that updates immediately when actuals come in.
    
    This is like having a coach watching every game and giving
    immediate feedback.
    """
    
    def __init__(self, db_client):
        self.db_client = db_client
        self.recent_errors = []
        self.error_buffer_size = 30
        
    def record_actual(self, date, actual_customers, actual_sales, actual_atv):
        """
        Called when actual data comes in. Immediately compares to forecast.
        """
        if self.db_client is None:
            return None
        
        try:
            # Get the forecast for this date
            date_str = date.strftime('%Y-%m-%d')
            forecast_doc = self.db_client.collection('forecast_log').document(date_str).get()
            
            if not forecast_doc.exists:
                return {'status': 'no_forecast', 'message': 'No forecast found for this date'}
            
            forecast = forecast_doc.to_dict()
            pred_customers = forecast.get('predicted_customers', 0)
            
            # Calculate error
            error = actual_customers - pred_customers
            pct_error = (error / actual_customers * 100) if actual_customers > 0 else 0
            
            # Store the error
            error_record = {
                'date': date,
                'predicted': pred_customers,
                'actual': actual_customers,
                'error': error,
                'pct_error': pct_error,
                'recorded_at': datetime.now()
            }
            
            # Save to Firestore
            self.db_client.collection('forecast_errors').document(date_str).set(error_record)
            
            # Immediate feedback
            feedback = self._generate_immediate_feedback(error_record)
            
            return {
                'status': 'recorded',
                'error': error_record,
                'feedback': feedback
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _generate_immediate_feedback(self, error_record):
        """Generate immediate feedback on the prediction."""
        pct_error = abs(error_record['pct_error'])
        direction = 'under' if error_record['error'] > 0 else 'over'
        
        if pct_error <= 5:
            grade = 'A'
            message = "Excellent prediction! Very close to actual."
        elif pct_error <= 10:
            grade = 'B'
            message = "Good prediction. Minor variance."
        elif pct_error <= 15:
            grade = 'C'
            message = f"Moderate accuracy. {direction.title()}-predicted by {pct_error:.1f}%"
        elif pct_error <= 20:
            grade = 'D'
            message = f"Below target accuracy. {direction.title()}-predicted by {pct_error:.1f}%"
        else:
            grade = 'F'
            message = f"Significant miss. {direction.title()}-predicted by {pct_error:.1f}%. Investigating..."
        
        return {
            'grade': grade,
            'message': message,
            'accuracy': 100 - pct_error
        }


class AdaptiveWeightManager:
    """
    Manages and adapts blend weights based on performance.
    
    Thinks like: "The SDLY component has been the most accurate lately.
    I should give it more weight until that changes."
    """
    
    def __init__(self, db_client):
        self.db_client = db_client
        self.default_weights = {
            'model': 0.50,
            'sdly_contextual': 0.25,
            'recent_trend': 0.25
        }
        self.weight_history = []
        
    def get_current_weights(self):
        """Get the current optimal blend weights."""
        if self.db_client is None:
            return self.default_weights
        
        try:
            doc = self.db_client.collection('ai_learning').document('blend_weights').get()
            if doc.exists:
                weights = doc.to_dict().get('weights')
                if weights and self._validate_weights(weights):
                    return weights
            return self.default_weights
        except:
            return self.default_weights
    
    def _validate_weights(self, weights):
        """Ensure weights are valid (sum to 1, all positive)."""
        required_keys = ['model', 'sdly_contextual', 'recent_trend']
        
        if not all(k in weights for k in required_keys):
            return False
        
        if not all(0 <= weights[k] <= 1 for k in required_keys):
            return False
        
        total = sum(weights[k] for k in required_keys)
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            return False
        
        return True
    
    def adapt_weights(self, component_performance):
        """
        Adapt weights based on recent component performance.
        
        component_performance: dict with keys 'model', 'sdly', 'recent' 
                              and values being MAPE percentages
        """
        # Convert MAPE to "goodness" (lower MAPE = better)
        total_inverse_mape = sum(1/mape for mape in component_performance.values() if mape > 0)
        
        if total_inverse_mape == 0:
            return self.default_weights
        
        # Calculate new weights proportional to inverse MAPE
        new_weights = {}
        mapping = {'model': 'model', 'sdly': 'sdly_contextual', 'recent': 'recent_trend'}
        
        for comp, mape in component_performance.items():
            if mape > 0:
                new_weights[mapping[comp]] = (1/mape) / total_inverse_mape
            else:
                new_weights[mapping[comp]] = self.default_weights[mapping[comp]]
        
        # Apply constraints: model should be at least 40%, others at least 10%
        new_weights['model'] = max(0.40, min(0.70, new_weights.get('model', 0.50)))
        remaining = 1 - new_weights['model']
        
        # Distribute remaining between sdly and recent
        sdly_ratio = new_weights.get('sdly_contextual', 0.25) / (new_weights.get('sdly_contextual', 0.25) + new_weights.get('recent_trend', 0.25))
        new_weights['sdly_contextual'] = remaining * sdly_ratio
        new_weights['recent_trend'] = remaining * (1 - sdly_ratio)
        
        # Ensure minimums
        for key in ['sdly_contextual', 'recent_trend']:
            if new_weights[key] < 0.10:
                deficit = 0.10 - new_weights[key]
                new_weights[key] = 0.10
                new_weights['model'] -= deficit
        
        return new_weights
