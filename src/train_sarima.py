import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import load_and_preprocess_data
from feature_engineering import perform_feature_engineering, get_train_val_split

def train_sarima():
    df = load_and_preprocess_data()
    df_features = perform_feature_engineering(df)
    train, valid = get_train_val_split(df_features)

    print("Training SARIMA model...")
    model = SARIMAX(
        train['Total'],
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    )
    results = model.fit()
    preds = results.forecast(steps=8)
    
    joblib.dump(results, "models/sarima.pkl")
    print("SARIMA model saved to models/sarima.pkl")
    
if __name__ == "__main__":
    train_sarima()
