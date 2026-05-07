import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow

from data_preprocessing import load_and_preprocess_data
from feature_engineering import perform_feature_engineering, get_train_val_split
from train_xgboost import FEATURES

def calculate_accuracy(y_true, y_pred):
    # Prevent division by zero
    y_true = np.where(y_true == 0, 1e-10, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    return max(0, accuracy) # Floor at 0%

def evaluate_sarima(valid):
    try:
        models = joblib.load("models/sarima.pkl")
        preds = []
        for state in valid['State'].unique():
            state_valid = valid[valid['State'] == state]
            model = models.get(state)
            if model is not None:
                preds.extend(model.forecast(steps=len(state_valid)).values)
            else:
                preds.extend(np.zeros(len(state_valid)))
        return np.array(preds)
    except Exception as e:
        print(f"Error evaluating SARIMA: {e}")
        return np.zeros(len(valid))

def evaluate_prophet(valid):
    try:
        models = joblib.load("models/prophet.pkl")
        preds = []
        for state in valid['State'].unique():
            state_valid = valid[valid['State'] == state]
            model = models.get(state)
            if model is not None:
                future = state_valid.reset_index()[['Date']]
                future.columns = ['ds']
                forecast = model.predict(future)
                preds.extend(forecast['yhat'].values)
            else:
                preds.extend(np.zeros(len(state_valid)))
        return np.array(preds)
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
        
        # Load same scaler structure
        df = load_and_preprocess_data()
        df_features = perform_feature_engineering(df)
        train, _ = get_train_val_split(df_features)
        scaler = MinMaxScaler()
        scaler.fit(train[['Total']])
        
        model = load_model("models/lstm.keras")
        preds_all = []
        window = 8
        
        for state in valid['State'].unique():
            state_train = train[train['State'] == state]
            state_valid = valid[valid['State'] == state]
            
            scaled_valid = scaler.transform(state_valid[['Total']])
            past_data = state_train[['Total']].tail(window).values
            scaled_past = scaler.transform(past_data)
            
            combined = np.vstack((scaled_past, scaled_valid))
            
            X = []
            for i in range(window, len(combined)):
                X.append(combined[i-window:i])
            
            if len(X) > 0:
                X_valid = np.array(X)
                preds = model.predict(X_valid, verbose=0)
                preds_inv = scaler.inverse_transform(preds)
                preds_all.extend(preds_inv.flatten())
            else:
                preds_all.extend(np.zeros(len(state_valid)))
                
        return np.array(preds_all)
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
    
    # Calculate RMSE (using np.sqrt because squared=False is removed in latest sklearn)
    rmse_sarima = np.sqrt(mean_squared_error(y_true, preds_sarima))
    rmse_prophet = np.sqrt(mean_squared_error(y_true, preds_prophet))
    rmse_xgboost = np.sqrt(mean_squared_error(y_true, preds_xgboost))
    rmse_lstm = np.sqrt(mean_squared_error(y_true, preds_lstm))
    
    results = {
        "sarima": rmse_sarima,
        "prophet": rmse_prophet,
        "xgboost": rmse_xgboost,
        "lstm": rmse_lstm
    }
    
    print("\n--- Evaluation Results (RMSE - Lower is Better) ---")
    for model, rmse in results.items():
        print(f"{model.capitalize()}: {rmse:.2f}")
        
    accuracies = {
        "sarima": calculate_accuracy(y_true, preds_sarima),
        "prophet": calculate_accuracy(y_true, preds_prophet),
        "xgboost": calculate_accuracy(y_true, preds_xgboost),
        "lstm": calculate_accuracy(y_true, preds_lstm)
    }
    
    print("\n--- Model Accuracies (Higher is Better) ---")
    for model, acc in accuracies.items():
        print(f"{model.capitalize()}: {acc:.2f}% Accuracy")
    
    
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
