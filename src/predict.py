import joblib
import pandas as pd
import numpy as np

def predict_next_8_weeks(state):
    # load model
    try:
        best_model_name = joblib.load("models/best_model_name.pkl")
        model = joblib.load(f"models/{best_model_name}.pkl")
    except:
        best_model_name = "xgboost"
        
    # fetch latest state data
    # build features
    # For production, we would load the last known features for the state and project them forward 8 weeks.
    
    # generate predictions
    # Simulating 8 week forecast based on the model for API demonstration purposes
    predictions = np.random.randint(50000, 200000, size=8).tolist()
    
    return predictions

if __name__ == "__main__":
    print(predict_next_8_weeks("Texas"))
