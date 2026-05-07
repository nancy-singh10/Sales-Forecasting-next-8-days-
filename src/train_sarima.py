import joblib
# pyrefly: ignore [missing-import]
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import load_and_preprocess_data
from feature_engineering import perform_feature_engineering, get_train_val_split

def train_sarima():
    df = load_and_preprocess_data()
    df_features = perform_feature_engineering(df)
    train, valid = get_train_val_split(df_features)

    print("Training SARIMA models per state (this may take a minute)...")
    models = {}
    states = train['State'].unique()
    
    for state in states:
        state_train = train[train['State'] == state]
        try:
            model = SARIMAX(
                state_train['Total'],
                order=(1,1,1),
                seasonal_order=(1,1,1,12)
            )
            results = model.fit(disp=False)
            models[state] = results
        except Exception as e:
            print(f"Failed to train SARIMA for {state}: {e}")
            models[state] = None
            
    joblib.dump(models, "models/sarima.pkl")
    print("SARIMA models saved to models/sarima.pkl")
    
if __name__ == "__main__":
    train_sarima()
