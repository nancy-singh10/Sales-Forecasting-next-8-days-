import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow

from data_preprocessing import load_and_preprocess_data
from feature_engineering import perform_feature_engineering, get_train_val_split
from train_xgboost import FEATURES

def evaluate_sarima(valid):
    try:
        model = joblib.load("models/sarima.pkl")
        preds = model.forecast(steps=len(valid))
        return preds.values
    except Exception as e:
        print(f"Error evaluating SARIMA: {e}")
        return np.zeros(len(valid))

def evaluate_prophet(valid):
    try:
        model = joblib.load("models/prophet.pkl")
        future = valid.reset_index()[['Date']]
        future.columns = ['ds']
        forecast = model.predict(future)
        return forecast['yhat'].values
    except Exception as e:
        print(f"Error evaluating Prophet: {e}")
        return np.zeros(len(valid))

def evaluate_xgboost(valid):
    try:
        model = joblib.load("models/xgboost.pkl")
        preds = model.predict(valid[FEATURES])
        return preds
    except Exception as e:
        print(f"Error evaluating XGBoost: {e}")
        return np.zeros(len(valid))

def evaluate_lstm(valid):
    try:
        from tensorflow.keras.models import load_model
        from sklearn.preprocessing import MinMaxScaler
        
        # Load same scaler structure (simplification for script)
        # Assuming training fitted scaler on train['Total']
        df = load_and_preprocess_data()
        df_features = perform_feature_engineering(df)
        train, _ = get_train_val_split(df_features)
        scaler = MinMaxScaler()
        scaler.fit(train[['Total']])
        
        scaled_valid = scaler.transform(valid[['Total']])
        X = []
        window = 8
        # We need past 8 days to predict, simplifying here by prepending train data
        past_data = train[['Total']].tail(window).values
        scaled_past = scaler.transform(past_data)
        
        combined = np.vstack((scaled_past, scaled_valid))
        
        for i in range(window, len(combined)):
            X.append(combined[i-window:i])
        
        X_valid = np.array(X)
        model = load_model("models/lstm.h5")
        preds = model.predict(X_valid)
        preds_inv = scaler.inverse_transform(preds)
        return preds_inv.flatten()
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")
        return np.zeros(len(valid))

def evaluate_models():
    df = load_and_preprocess_data()
    df_features = perform_feature_engineering(df)
    _, valid = get_train_val_split(df_features)
    
    y_true = valid['Total'].values
    
    preds_sarima = evaluate_sarima(valid)
    preds_prophet = evaluate_prophet(valid)
    preds_xgboost = evaluate_xgboost(valid)
    preds_lstm = evaluate_lstm(valid)
    
    # Calculate RMSE
    rmse_sarima = mean_squared_error(y_true, preds_sarima, squared=False)
    rmse_prophet = mean_squared_error(y_true, preds_prophet, squared=False)
    rmse_xgboost = mean_squared_error(y_true, preds_xgboost, squared=False)
    rmse_lstm = mean_squared_error(y_true, preds_lstm, squared=False)
    
    results = {
        "sarima": rmse_sarima,
        "prophet": rmse_prophet,
        "xgboost": rmse_xgboost,
        "lstm": rmse_lstm
    }
    
    print("Evaluation Results (RMSE):", results)
    
    best_model = min(results, key=results.get)
    print(f"Best Model selected: {best_model}")
    
    joblib.dump(best_model, "models/best_model_name.pkl")
    
    # MLFlow Tracking
    try:
        mlflow.set_experiment("Sales Forecasting")
        with mlflow.start_run():
            mlflow.log_metrics(results)
            mlflow.log_param("best_model", best_model)
    except Exception as e:
        print(f"MLflow not started or error: {e}")

if __name__ == "__main__":
    evaluate_models()
