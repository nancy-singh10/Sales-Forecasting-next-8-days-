import joblib
from xgboost import XGBRegressor

from data_preprocessing import load_and_preprocess_data
from feature_engineering import perform_feature_engineering, get_train_val_split

FEATURES = [
    'lag_1',
    'lag_7',
    'lag_30',
    'rolling_mean_7',
    'rolling_std_7',
    'month',
    'week',
    'quarter',
    'year',
    'is_holiday'
]

def train_xgboost():
    df = load_and_preprocess_data()
    df_features = perform_feature_engineering(df)
    train, valid = get_train_val_split(df_features)

    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5
    )
    model.fit(
        train[FEATURES],
        train['Total']
    )
    
    preds = model.predict(valid[FEATURES])
    
    joblib.dump(model, "models/xgboost.pkl")
    print("XGBoost model saved to models/xgboost.pkl")

if __name__ == "__main__":
    train_xgboost()
