import joblib
import pandas as pd
import numpy as np

from data_preprocessing import load_and_preprocess_data
from feature_engineering import perform_feature_engineering, get_train_val_split
from train_xgboost import FEATURES

def predict_next_8_weeks(state):
    try:
        best_model_name = joblib.load("models/best_model_name.pkl")
    except:
        best_model_name = "xgboost"
    
    # Load the dataset and build features
    df = load_and_preprocess_data()
    df_features = perform_feature_engineering(df)
    
    # Filter to the requested state
    state_data = df_features[df_features['State'] == state].copy()
    
    if len(state_data) == 0:
        return {"error": f"State '{state}' not found in dataset"}
    
    if best_model_name == "xgboost":
        model = joblib.load("models/xgboost.pkl")
        # Use the last row's features as starting point and forecast iteratively
        predictions = []
        latest = state_data.tail(1).copy()
        
        for week in range(8):
            pred = model.predict(latest[FEATURES])[0]
            predictions.append(round(float(pred), 2))
            
            # Shift features forward for next prediction
            latest['lag_30'] = latest['lag_7'].values[0]
            latest['lag_7'] = latest['lag_1'].values[0]
            latest['lag_1'] = pred
            latest['rolling_mean_7'] = (latest['rolling_mean_7'].values[0] * 6 + pred) / 7
            latest['week'] = latest['week'].values[0] + 1
        
        return predictions
    
    elif best_model_name in ["sarima", "prophet"]:
        models_dict = joblib.load(f"models/{best_model_name}.pkl")
        model = models_dict.get(state)
        
        if model is None:
            return {"error": f"No {best_model_name} model found for {state}"}
        
        if best_model_name == "sarima":
            preds = model.forecast(steps=8)
            return [round(float(p), 2) for p in preds.values]
        else:
            future = model.make_future_dataframe(periods=8, freq='W')
            forecast = model.predict(future)
            return [round(float(p), 2) for p in forecast['yhat'].tail(8).values]
    
    else:
        return {"error": f"Unsupported model: {best_model_name}"}

if __name__ == "__main__":
    print("Predicting next 8 weeks for Texas...")
    print(predict_next_8_weeks("Texas"))
