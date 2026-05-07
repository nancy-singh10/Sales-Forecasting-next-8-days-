import joblib
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import load_and_preprocess_data
from feature_engineering import perform_feature_engineering, get_train_val_split

def train_prophet_model():
    df = load_and_preprocess_data()
    df_features = perform_feature_engineering(df)
    train, valid = get_train_val_split(df_features)

    print("Training Prophet models per state (this may take a minute)...")
    models = {}
    states = train['State'].unique()
    
    for state in states:
        state_train = train[train['State'] == state]
        prophet_df = state_train.reset_index()[['Date', 'Total']]
        prophet_df.columns = ['ds', 'y']
        
        try:
            model = Prophet()
            model.fit(prophet_df)
            models[state] = model
        except Exception as e:
            print(f"Failed to train Prophet for {state}: {e}")
            models[state] = None
            
    joblib.dump(models, "models/prophet.pkl")
    print("Prophet models saved to models/prophet.pkl")

if __name__ == "__main__":
    train_prophet_model()
