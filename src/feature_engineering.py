import pandas as pd
import holidays

def perform_feature_engineering(df):
    # Lag Features
    df['lag_1'] = df['Total'].shift(1)
    df['lag_7'] = df['Total'].shift(7)
    df['lag_30'] = df['Total'].shift(30)
    
    # Rolling Mean
    df['rolling_mean_7'] = df['Total'].rolling(7).mean()
    
    # Rolling Standard Deviation
    df['rolling_std_7'] = df['Total'].rolling(7).std()
    
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
    train = df[:-8]
    valid = df[-8:]
    return train, valid

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    df = load_and_preprocess_data()
    df_features = perform_feature_engineering(df)
    train, valid = get_train_val_split(df_features)
    print("Train shape:", train.shape)
    print("Valid shape:", valid.shape)
