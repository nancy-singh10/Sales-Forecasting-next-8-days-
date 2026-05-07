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

    print("Training Prophet model...")
    prophet_df = train.reset_index()[['Date', 'Total']]
    prophet_df.columns = ['ds', 'y']
    
    model = Prophet()
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=8, freq='W')
    forecast = model.predict(future)
    
    joblib.dump(model, "models/prophet.pkl")
    print("Prophet model saved to models/prophet.pkl")

if __name__ == "__main__":
    train_prophet_model()
