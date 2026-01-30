# trend_intelligence.py - TREND-AWARE FORECASTING v11.1
# 
# This module fixes the over-prediction problem by:
# 1. Detecting trend DIRECTION from multiple sources
# 2. Confirming trends when SDLY and Recent agree
# 3. Dynamically adjusting blend weights based on trend strength
# 4. Applying momentum dampening when weakness is detected
# 5. Using conservative estimates during uncertain periods

import pandas as pd
import numpy as np
from datetime import timedelta


class TrendAnalyzer:
    """
    Deep analysis of trend direction and strength.
    
    Thinks like: "Let me look at ALL the signals and see if they're
    telling me the same story. If both last year and recent weeks
    show weakness, I should BELIEVE that weakness."
    """
    
    def __init__(self):
        self.trend_signals = {}
        self.confidence_in_trend = 0
        
    def analyze_comprehensive_trend(self, historical_df, target_date, events_df=None):
        """
        Comprehensive trend analysis that looks at multiple timeframes
        and confirms trends across different data sources.
        """
        signals = {
            'sdly_trend': self._analyze_sdly_trend(historical_df, target_date),
            'recent_trend': self._analyze_recent_trend(historical_df, target_date),
            'weekly_momentum': self._analyze_weekly_momentum(historical_df, target_date),
            'dow_trend': self._analyze_dow_trend(historical_df, target_date),
            'monthly_pattern': self._analyze_monthly_pattern(historical_df, target_date)
        }
        
        # Determine overall trend direction
        overall = self._synthesize_trends(signals)
        
        return {
            'signals': signals,
            'overall_direction': overall['direction'],
            'overall_strength': overall['strength'],
            'confidence': overall['confidence'],
            'recommended_adjustment': overall['adjustment'],
            'reasoning': overall['reasoning']
        }
    
    def _analyze_sdly_trend(self, df, target_date):
        """
        Analyze what happened in the same period last year.
        Look at the 14 days BEFORE and AFTER the SDLY date.
        """
        sdly_date = target_date - timedelta(days=364)
        
        # Get 14 days before SDLY
        before_start = sdly_date - timedelta(days=14)
        before_end = sdly_date - timedelta(days=1)
        before_data = df[(df['date'] >= before_start) & (df['date'] <= before_end)]
        
        # Get 14 days after SDLY (including SDLY)
        after_start = sdly_date
        after_end = sdly_date + timedelta(days=14)
        after_data = df[(df['date'] >= after_start) & (df['date'] <= after_end)]
        
        if before_data.empty or after_data.empty:
            return {'direction': 'neutral', 'strength': 0, 'data_available': False}
        
        before_mean = before_data['customers'].mean()
        after_mean = after_data['customers'].mean()
        
        # Calculate trend
        change_pct = (after_mean - before_mean) / before_mean * 100
        
        # Also look at the slope within the after period
        if len(after_data) >= 5:
            after_data = after_data.sort_values('date')
            x = np.arange(len(after_data))
            y = after_data['customers'].values
            slope = np.polyfit(x, y, 1)[0]
            daily_change_pct = slope / after_mean * 100
        else:
            daily_change_pct = 0
        
        # Determine direction
        if change_pct < -5 or daily_change_pct < -0.5:
            direction = 'down'
            strength = min(abs(change_pct) / 20, 1.0)  # Normalize to 0-1
        elif change_pct > 5 or daily_change_pct > 0.5:
            direction = 'up'
            strength = min(abs(change_pct) / 20, 1.0)
        else:
            direction = 'neutral'
            strength = 0
        
        return {
            'direction': direction,
            'strength': strength,
            'change_pct': change_pct,
            'daily_slope_pct': daily_change_pct,
            'before_mean': before_mean,
            'after_mean': after_mean,
            'data_available': True
        }
    
    def _analyze_recent_trend(self, df, target_date):
        """
        Analyze the most recent weeks with emphasis on the last 2 weeks.
        """
        end_date = df['date'].max()
        
        # Last 2 weeks (most important)
        last_2w_start = end_date - timedelta(days=14)
        last_2w = df[(df['date'] > last_2w_start) & (df['date'] <= end_date)]
        
        # Previous 2 weeks (for comparison)
        prev_2w_start = end_date - timedelta(days=28)
        prev_2w_end = end_date - timedelta(days=14)
        prev_2w = df[(df['date'] > prev_2w_start) & (df['date'] <= prev_2w_end)]
        
        # Weeks 5-8 (baseline)
        baseline_start = end_date - timedelta(days=56)
        baseline_end = end_date - timedelta(days=28)
        baseline = df[(df['date'] > baseline_start) & (df['date'] <= baseline_end)]
        
        if last_2w.empty or prev_2w.empty:
            return {'direction': 'neutral', 'strength': 0, 'data_available': False}
        
        last_2w_mean = last_2w['customers'].mean()
        prev_2w_mean = prev_2w['customers'].mean()
        baseline_mean = baseline['customers'].mean() if not baseline.empty else prev_2w_mean
        
        # Week-over-week change
        wow_change = (last_2w_mean - prev_2w_mean) / prev_2w_mean * 100
        
        # Vs baseline change
        vs_baseline_change = (last_2w_mean - baseline_mean) / baseline_mean * 100 if baseline_mean > 0 else 0
        
        # Calculate slope within last 2 weeks
        if len(last_2w) >= 5:
            last_2w = last_2w.sort_values('date')
            x = np.arange(len(last_2w))
            y = last_2w['customers'].values
            slope = np.polyfit(x, y, 1)[0]
            daily_slope_pct = slope / last_2w_mean * 100
        else:
            daily_slope_pct = 0
        
        # Determine direction - be MORE sensitive to recent weakness
        if wow_change < -3 or daily_slope_pct < -0.3:
            direction = 'down'
            # Strength is higher when multiple indicators agree
            strength = min((abs(wow_change) + abs(daily_slope_pct) * 5) / 25, 1.0)
        elif wow_change > 3 or daily_slope_pct > 0.3:
            direction = 'up'
            strength = min((abs(wow_change) + abs(daily_slope_pct) * 5) / 25, 1.0)
        else:
            direction = 'neutral'
            strength = 0
        
        return {
            'direction': direction,
            'strength': strength,
            'wow_change_pct': wow_change,
            'vs_baseline_pct': vs_baseline_change,
            'daily_slope_pct': daily_slope_pct,
            'last_2w_mean': last_2w_mean,
            'prev_2w_mean': prev_2w_mean,
            'data_available': True
        }
    
    def _analyze_weekly_momentum(self, df, target_date):
        """
        Week-by-week momentum analysis for the last 4 weeks.
        """
        end_date = df['date'].max()
        
        weekly_avgs = []
        for w in range(4):
            week_start = end_date - timedelta(days=(w + 1) * 7)
            week_end = end_date - timedelta(days=w * 7)
            week_data = df[(df['date'] > week_start) & (df['date'] <= week_end)]
            if not week_data.empty:
                weekly_avgs.append({
                    'week': w + 1,
                    'avg': week_data['customers'].mean()
                })
        
        if len(weekly_avgs) < 3:
            return {'direction': 'neutral', 'strength': 0, 'data_available': False}
        
        # Check if consecutive weeks are declining
        declining_weeks = 0
        for i in range(len(weekly_avgs) - 1):
            if weekly_avgs[i]['avg'] < weekly_avgs[i + 1]['avg']:
                declining_weeks += 1
        
        # Check if consecutive weeks are increasing
        increasing_weeks = 0
        for i in range(len(weekly_avgs) - 1):
            if weekly_avgs[i]['avg'] > weekly_avgs[i + 1]['avg']:
                increasing_weeks += 1
        
        # Calculate overall momentum
        if len(weekly_avgs) >= 2:
            recent_vs_older = (weekly_avgs[0]['avg'] - weekly_avgs[-1]['avg']) / weekly_avgs[-1]['avg'] * 100
        else:
            recent_vs_older = 0
        
        if declining_weeks >= 2 or recent_vs_older < -5:
            direction = 'down'
            strength = min(declining_weeks / 3 + abs(recent_vs_older) / 20, 1.0)
        elif increasing_weeks >= 2 or recent_vs_older > 5:
            direction = 'up'
            strength = min(increasing_weeks / 3 + abs(recent_vs_older) / 20, 1.0)
        else:
            direction = 'neutral'
            strength = 0
        
        return {
            'direction': direction,
            'strength': strength,
            'declining_weeks': declining_weeks,
            'increasing_weeks': increasing_weeks,
            'recent_vs_older_pct': recent_vs_older,
            'weekly_avgs': weekly_avgs,
            'data_available': True
        }
    
    def _analyze_dow_trend(self, df, target_date):
        """
        Analyze if the specific day of week is trending differently.
        """
        target_dow = target_date.weekday()
        dow_data = df[df['date'].dt.dayofweek == target_dow].copy()
        
        if len(dow_data) < 8:
            return {'direction': 'neutral', 'strength': 0, 'data_available': False}
        
        dow_data = dow_data.sort_values('date')
        
        # Last 4 occurrences vs previous 4
        recent_4 = dow_data.tail(4)['customers'].mean()
        prev_4 = dow_data.tail(8).head(4)['customers'].mean()
        
        change_pct = (recent_4 - prev_4) / prev_4 * 100
        
        if change_pct < -5:
            direction = 'down'
            strength = min(abs(change_pct) / 15, 1.0)
        elif change_pct > 5:
            direction = 'up'
            strength = min(abs(change_pct) / 15, 1.0)
        else:
            direction = 'neutral'
            strength = 0
        
        return {
            'direction': direction,
            'strength': strength,
            'change_pct': change_pct,
            'recent_4_avg': recent_4,
            'prev_4_avg': prev_4,
            'data_available': True
        }
    
    def _analyze_monthly_pattern(self, df, target_date):
        """
        Analyze if this part of the month typically shows a pattern.
        """
        target_day = target_date.day
        
        # Get data from same part of month historically
        similar_days = df[df['date'].dt.day.between(target_day - 3, target_day + 3)]
        
        if len(similar_days) < 10:
            return {'direction': 'neutral', 'strength': 0, 'data_available': False}
        
        # Compare recent months to older months
        cutoff = df['date'].max() - timedelta(days=90)
        recent_similar = similar_days[similar_days['date'] > cutoff]
        older_similar = similar_days[similar_days['date'] <= cutoff]
        
        if recent_similar.empty or older_similar.empty:
            return {'direction': 'neutral', 'strength': 0, 'data_available': False}
        
        recent_mean = recent_similar['customers'].mean()
        older_mean = older_similar['customers'].mean()
        
        change_pct = (recent_mean - older_mean) / older_mean * 100
        
        if change_pct < -5:
            direction = 'down'
            strength = min(abs(change_pct) / 20, 1.0)
        elif change_pct > 5:
            direction = 'up'
            strength = min(abs(change_pct) / 20, 1.0)
        else:
            direction = 'neutral'
            strength = 0
        
        return {
            'direction': direction,
            'strength': strength,
            'change_pct': change_pct,
            'data_available': True
        }
    
    def _synthesize_trends(self, signals):
        """
        Synthesize all trend signals into an overall assessment.
        
        KEY INSIGHT: When multiple signals agree on direction,
        we should have HIGH confidence and adjust more aggressively.
        """
        directions = []
        strengths = []
        weights = {
            'sdly_trend': 2.0,       # Historical pattern is important
            'recent_trend': 3.0,     # Recent trend is MOST important
            'weekly_momentum': 2.5,  # Week-over-week momentum matters
            'dow_trend': 1.5,        # Day-specific trends
            'monthly_pattern': 1.0   # Monthly patterns (least weight)
        }
        
        reasoning = []
        
        for signal_name, signal_data in signals.items():
            if signal_data.get('data_available', False):
                direction = signal_data['direction']
                strength = signal_data['strength']
                weight = weights.get(signal_name, 1.0)
                
                directions.append((direction, strength, weight))
                
                if direction == 'down' and strength > 0.3:
                    reasoning.append(f"{signal_name}: DOWN (strength {strength:.2f})")
                elif direction == 'up' and strength > 0.3:
                    reasoning.append(f"{signal_name}: UP (strength {strength:.2f})")
        
        if not directions:
            return {
                'direction': 'neutral',
                'strength': 0,
                'confidence': 0,
                'adjustment': 1.0,
                'reasoning': 'Insufficient data for trend analysis'
            }
        
        # Count weighted votes for each direction
        up_score = sum(s * w for d, s, w in directions if d == 'up')
        down_score = sum(s * w for d, s, w in directions if d == 'down')
        total_weight = sum(w for _, _, w in directions)
        
        # Calculate agreement level
        up_signals = sum(1 for d, s, _ in directions if d == 'up' and s > 0.2)
        down_signals = sum(1 for d, s, _ in directions if d == 'down' and s > 0.2)
        total_signals = len([d for d, s, _ in directions if s > 0.2])
        
        # Determine overall direction
        if down_score > up_score and down_score / total_weight > 0.3:
            overall_direction = 'down'
            overall_strength = down_score / total_weight
            agreement = down_signals / total_signals if total_signals > 0 else 0
        elif up_score > down_score and up_score / total_weight > 0.3:
            overall_direction = 'up'
            overall_strength = up_score / total_weight
            agreement = up_signals / total_signals if total_signals > 0 else 0
        else:
            overall_direction = 'neutral'
            overall_strength = 0
            agreement = 0
        
        # Calculate confidence (higher when signals agree)
        confidence = min(agreement * overall_strength * 100, 100)
        
        # Calculate recommended adjustment
        # KEY: When trends are DOWN, we should REDUCE predictions
        if overall_direction == 'down':
            # More aggressive reduction when confidence is high
            reduction = overall_strength * (0.5 + agreement * 0.5)  # Up to 100% of strength
            adjustment = 1 - (reduction * 0.15)  # Max 15% reduction
            adjustment = max(0.80, adjustment)  # Floor at 20% reduction
            
            reasoning.append(f"\n>>> TREND CONFIRMED: {down_signals}/{total_signals} signals show WEAKNESS")
            reasoning.append(f">>> Recommended adjustment: {(1-adjustment)*100:.1f}% REDUCTION")
        elif overall_direction == 'up':
            increase = overall_strength * (0.5 + agreement * 0.5)
            adjustment = 1 + (increase * 0.10)  # Max 10% increase (more conservative on upside)
            adjustment = min(1.15, adjustment)
            
            reasoning.append(f"\n>>> TREND CONFIRMED: {up_signals}/{total_signals} signals show STRENGTH")
            reasoning.append(f">>> Recommended adjustment: {(adjustment-1)*100:.1f}% INCREASE")
        else:
            adjustment = 1.0
            reasoning.append("\n>>> No clear trend direction - maintaining baseline forecast")
        
        return {
            'direction': overall_direction,
            'strength': overall_strength,
            'confidence': confidence,
            'adjustment': adjustment,
            'agreement': agreement,
            'reasoning': '\n'.join(reasoning)
        }


class TrendAwareBlender:
    """
    Blends predictions with awareness of trend direction.
    
    Key insight: When all signals show weakness, we should:
    1. Trust the LOWER predictions more
    2. Apply momentum dampening
    3. Use conservative estimates
    """
    
    def __init__(self):
        self.base_weights = {
            'model': 0.40,
            'sdly': 0.30,
            'recent': 0.30
        }
        
    def blend_with_trend_awareness(self, model_pred, sdly_pred, recent_pred, trend_analysis):
        """
        Blend predictions with trend-aware adjustments.
        """
        direction = trend_analysis['overall_direction']
        strength = trend_analysis['overall_strength']
        confidence = trend_analysis['confidence']
        adjustment = trend_analysis['recommended_adjustment']
        
        # Start with base weights
        weights = self.base_weights.copy()
        
        # Adjust weights based on trend direction
        if direction == 'down' and confidence > 30:
            # When trending down, trust the LOWER predictions more
            predictions = {'model': model_pred, 'sdly': sdly_pred, 'recent': recent_pred}
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1])
            
            # Give more weight to the lowest prediction
            lowest_key = sorted_preds[0][0]
            weights[lowest_key] += 0.15 * (confidence / 100)
            
            # Reduce weight of highest prediction
            highest_key = sorted_preds[-1][0]
            weights[highest_key] -= 0.15 * (confidence / 100)
            
            # Ensure weights stay valid
            weights = {k: max(0.1, v) for k, v in weights.items()}
            
        elif direction == 'up' and confidence > 30:
            # When trending up, give slight boost to higher predictions
            # But be more conservative (don't want to over-predict)
            predictions = {'model': model_pred, 'sdly': sdly_pred, 'recent': recent_pred}
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            highest_key = sorted_preds[0][0]
            weights[highest_key] += 0.10 * (confidence / 100)
            
            lowest_key = sorted_preds[-1][0]
            weights[lowest_key] -= 0.10 * (confidence / 100)
            
            weights = {k: max(0.1, v) for k, v in weights.items()}
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        # Calculate blended prediction
        blended = (
            weights['model'] * model_pred +
            weights['sdly'] * sdly_pred +
            weights['recent'] * recent_pred
        )
        
        # Apply trend adjustment
        adjusted = blended * adjustment
        
        return {
            'blended_raw': blended,
            'trend_adjusted': adjusted,
            'adjustment_applied': adjustment,
            'weights_used': weights,
            'trend_direction': direction,
            'trend_confidence': confidence
        }


class MomentumDampener:
    """
    Applies momentum dampening when recent trends show weakness.
    
    Thinks like: "The recent trend is clearly down. Even if my model
    thinks it should be higher, I need to respect the momentum."
    """
    
    def __init__(self):
        self.dampening_factor = 0.8  # How much to dampen (0-1)
        
    def calculate_dampening(self, historical_df, target_date, base_prediction):
        """
        Calculate how much to dampen the prediction based on momentum.
        """
        end_date = historical_df['date'].max()
        
        # Get last 14 days average
        last_14_start = end_date - timedelta(days=14)
        last_14 = historical_df[historical_df['date'] > last_14_start]
        
        if last_14.empty:
            return base_prediction, 1.0, "No recent data for dampening"
        
        recent_avg = last_14['customers'].mean()
        
        # If prediction is significantly higher than recent average, dampen it
        if base_prediction > recent_avg * 1.10:  # More than 10% above recent
            excess = (base_prediction - recent_avg) / recent_avg
            dampening = 1 - (excess * self.dampening_factor * 0.5)
            dampening = max(0.85, dampening)  # Don't dampen more than 15%
            
            dampened_prediction = base_prediction * dampening
            
            return (
                dampened_prediction,
                dampening,
                f"Prediction {excess*100:.1f}% above recent avg - dampened by {(1-dampening)*100:.1f}%"
            )
        
        return base_prediction, 1.0, "No dampening needed"


class ConservativeEstimator:
    """
    Provides conservative estimates during uncertain or declining periods.
    
    Thinks like: "Given the uncertainty, I should provide a range
    and lean toward the conservative side for planning purposes."
    """
    
    def __init__(self):
        pass
    
    def get_conservative_estimate(self, predictions_dict, trend_analysis):
        """
        Get a conservative estimate based on multiple predictions.
        """
        model = predictions_dict.get('model', 0)
        sdly = predictions_dict.get('sdly', 0)
        recent = predictions_dict.get('recent', 0)
        
        values = [v for v in [model, sdly, recent] if v > 0]
        
        if not values:
            return None
        
        direction = trend_analysis.get('overall_direction', 'neutral')
        confidence = trend_analysis.get('confidence', 0)
        
        # Calculate statistics
        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        if direction == 'down' and confidence > 40:
            # During downtrend, lean toward the lower estimate
            # Weight: 60% toward minimum, 40% toward mean
            conservative = min_val * 0.6 + mean_val * 0.4
            reasoning = "Downtrend detected - using conservative (lower) estimate"
        elif direction == 'up' and confidence > 40:
            # During uptrend, use mean (don't get too optimistic)
            conservative = mean_val
            reasoning = "Uptrend detected - using mean estimate (staying cautious)"
        else:
            # Neutral - use slightly conservative mean
            conservative = mean_val * 0.95
            reasoning = "Neutral trend - using slightly conservative estimate"
        
        return {
            'conservative_estimate': int(round(conservative)),
            'optimistic_estimate': int(round(max_val)),
            'pessimistic_estimate': int(round(min_val)),
            'mean_estimate': int(round(mean_val)),
            'reasoning': reasoning
        }


def apply_trend_intelligence(historical_df, target_date, model_pred, sdly_pred, recent_pred):
    """
    Main function to apply trend-aware intelligence to predictions.
    
    This is what makes the forecaster "think deeply" about trends.
    """
    # Initialize components
    trend_analyzer = TrendAnalyzer()
    blender = TrendAwareBlender()
    dampener = MomentumDampener()
    estimator = ConservativeEstimator()
    
    # Step 1: Analyze trends comprehensively
    trend_analysis = trend_analyzer.analyze_comprehensive_trend(historical_df, target_date)
    
    # Step 2: Blend with trend awareness
    blended = blender.blend_with_trend_awareness(
        model_pred, sdly_pred, recent_pred, trend_analysis
    )
    
    # Step 3: Apply momentum dampening if needed
    dampened, dampening_factor, dampening_reason = dampener.calculate_dampening(
        historical_df, target_date, blended['trend_adjusted']
    )
    
    # Step 4: Get conservative estimate
    conservative = estimator.get_conservative_estimate(
        {'model': model_pred, 'sdly': sdly_pred, 'recent': recent_pred},
        trend_analysis
    )
    
    # Final prediction: use the dampened value but cap at conservative estimate if trending down
    if trend_analysis['overall_direction'] == 'down' and trend_analysis['confidence'] > 50:
        final_prediction = min(dampened, conservative['conservative_estimate'])
        selection_reason = "Used minimum of dampened and conservative (strong downtrend)"
    else:
        final_prediction = dampened
        selection_reason = "Used dampened prediction"
    
    return {
        'final_prediction': int(round(final_prediction)),
        'trend_analysis': trend_analysis,
        'blending': blended,
        'dampening': {
            'factor': dampening_factor,
            'reason': dampening_reason,
            'dampened_value': dampened
        },
        'conservative_estimates': conservative,
        'selection_reason': selection_reason,
        'reasoning_summary': _generate_reasoning_summary(
            trend_analysis, blended, dampening_reason, selection_reason,
            model_pred, sdly_pred, recent_pred, final_prediction
        )
    }


def _generate_reasoning_summary(trend_analysis, blended, dampening_reason, selection_reason,
                                model_pred, sdly_pred, recent_pred, final_pred):
    """
    Generate a human-readable summary of the AI's reasoning.
    """
    lines = []
    lines.append("=" * 50)
    lines.append("🧠 AI TREND-AWARE REASONING")
    lines.append("=" * 50)
    
    lines.append("\n📊 INPUT PREDICTIONS:")
    lines.append(f"   • Model prediction: {model_pred:,.0f}")
    lines.append(f"   • SDLY prediction: {sdly_pred:,.0f}")
    lines.append(f"   • Recent trend prediction: {recent_pred:,.0f}")
    
    lines.append("\n📈 TREND ANALYSIS:")
    lines.append(f"   • Overall direction: {trend_analysis['overall_direction'].upper()}")
    lines.append(f"   • Confidence: {trend_analysis['confidence']:.0f}%")
    lines.append(f"   • Adjustment factor: {trend_analysis['recommended_adjustment']:.3f}")
    
    lines.append("\n🔍 SIGNAL DETAILS:")
    for signal_name, signal_data in trend_analysis['signals'].items():
        if signal_data.get('data_available'):
            direction = signal_data.get('direction', 'neutral')
            strength = signal_data.get('strength', 0)
            lines.append(f"   • {signal_name}: {direction.upper()} (strength: {strength:.2f})")
    
    lines.append("\n⚖️ BLENDING:")
    lines.append(f"   • Weights used: {blended['weights_used']}")
    lines.append(f"   • Raw blended: {blended['blended_raw']:,.0f}")
    lines.append(f"   • After trend adjustment: {blended['trend_adjusted']:,.0f}")
    
    lines.append(f"\n🎯 DAMPENING:")
    lines.append(f"   • {dampening_reason}")
    
    lines.append(f"\n✅ FINAL DECISION:")
    lines.append(f"   • {selection_reason}")
    lines.append(f"   • FINAL PREDICTION: {final_pred:,}")
    
    diff_from_model = (final_pred - model_pred) / model_pred * 100
    lines.append(f"   • Adjustment from model: {diff_from_model:+.1f}%")
    
    lines.append("\n" + "=" * 50)
    
    return "\n".join(lines)
