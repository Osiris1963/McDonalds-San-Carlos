# forecasting.py
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.statespace.sarima import SARIMAX
from data_processing import create_advanced_features

def train_customer_model_lgbm(historical_df, events_df):
    """
    Trains the powerful LightGBM model for customer forecasting using the original recursive strategy.
    """
    df_featured = create_advanced_features(historical_df, events_df)
    train_df = df_featured.dropna(subset=['sales_rolling_mean_14']).reset_index(drop=True)

    FEATURES = [col for col in train_df.columns if col not in [
        'date', 'sales', 'customers', 'atv', 'doc_id', 'day_type', 'day_type_notes', 'base_sales'
    ]]
    TARGET_CUST = 'customers'

    lgbm_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(train_df[FEATURES], train_df[TARGET_CUST])
    return model

def train_atv_model_sarima(historical_df):
    """
    Trains a robust SARIMA model for base_sales forecasting.
    This model excels at capturing seasonal and trend patterns.
    """
    # SARIMA works best on a clean time series index
    df = historical_df.set_index('date').copy()
    df['base_sales'] = df['sales'] - df.get('add_on_sales', 0)
    
    # Standard SARIMA parameters for weekly seasonality (s=7)
    # (p,d,q) handle non-seasonal components; (P,D,Q,s) handle seasonality.
    # These are sensible defaults that can be fine-tuned later if needed.
    model = SARIMAX(
        df['base_sales'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    return results

def generate_forecast_from_models(customer_model, atv_model, historical_df, events_df, periods=15):
    """
    Generates a forecast using the two pre-trained specialist models.
    """
    # Get all base_sales/ATV predictions from the SARIMA model at once.
    atv_forecast_results = atv_model.get_forecast(steps=periods)
    predicted_base_sales = atv_forecast_results.predicted_mean

    # Use the recursive strategy for the LightGBM customer model.
    customer_predictions = []
    history_df = historical_df.copy()
    last_date = history_df['date'].max()

    for i in range(periods):
        current_pred_date = last_date + timedelta(days=i + 1)
        
        future_placeholder = pd.DataFrame([{'date': current_pred_date}])
        temp_df = pd.concat([history_df, future_placeholder], ignore_index=True)
        featured_for_pred = create_advanced_features(temp_df, events_df)
        X_pred = featured_for_pred[[col for col in featured_for_pred.columns if col in customer_model.feature_name_]].iloc[-1:]

        pred_cust = customer_model.predict(X_pred)[0]
        customer_predictions.append(pred_cust)

        # Append predicted row to history for next loop iteration
        new_row = {'date': current_pred_date, 'customers': pred_cust, 'sales': 0, 'add_on_sales': 0}
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)

    # Combine the forecasts
    avg_addon_per_customer = historical_df['add_on_sales'].sum() / historical_df['customers'].replace(0, 1).sum()
    final_predictions = []
    for i in range(periods):
        pred_cust = max(0, customer_predictions[i])
        pred_base_sales = max(0, predicted_base_sales.iloc[i])
        
        pred_atv = (pred_base_sales / pred_cust) if pred_cust > 0 else 0
        estimated_add_on_sales = pred_cust * avg_addon_per_customer
        pred_total_sales = pred_base_sales + estimated_add_on_sales

        final_predictions.append({
            'ds': last_date + timedelta(days=i + 1),
            'forecast_customers': pred_cust,
            'forecast_atv': pred_atv,
            'forecast_sales': pred_total_sales
        })
        
    final_forecast = pd.DataFrame(final_predictions)
    final_forecast['forecast_customers'] = final_forecast['forecast_customers'].round().astype(int)

    return final_forecast
