import pandas as pd
import holidays

def perform_feature_engineering(df):
    # Lag Features (grouped by state to prevent leakage between states)
    df['lag_1'] = df.groupby('State')['Total'].shift(1)
    df['lag_7'] = df.groupby('State')['Total'].shift(7)
    df['lag_30'] = df.groupby('State')['Total'].shift(30)
    
    # Rolling Mean & Std (grouped by state)
    df['rolling_mean_7'] = df.groupby('State')['Total'].transform(lambda x: x.rolling(7).mean())
    df['rolling_std_7'] = df.groupby('State')['Total'].transform(lambda x: x.rolling(7).std())
    
    # Date Features
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week.astype(int)
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Holiday Features
    us_holidays = holidays.US()
    df['is_holiday'] = df.index.map(lambda x: 1 if x in us_holidays else 0)
    
    # Remove Nulls
    df = df.dropna()
    
    return df

def get_train_val_split(df):
    # NO random split. Time-series must preserve chronology.
    # Since we have 50 states stacked, we must use a date cutoff instead of just df[:-8]
    # to ensure we get the last 8 weeks for ALL states.
    unique_dates = sorted(df.index.unique())
    cutoff_date = unique_dates[-8]
    
    train = df[df.index < cutoff_date]
    valid = df[df.index >= cutoff_date]
    return train, valid

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    df = load_and_preprocess_data()
    df_features = perform_feature_engineering(df)
    train, valid = get_train_val_split(df_features)
    print("Train shape:", train.shape)
    print("Valid shape:", valid.shape)
