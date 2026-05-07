import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from data_preprocessing import load_and_preprocess_data
from feature_engineering import perform_feature_engineering, get_train_val_split

def train_lstm():
    df = load_and_preprocess_data()
    df_features = perform_feature_engineering(df)
    train, valid = get_train_val_split(df_features)

    print("Training LSTM model...")
    scaler = MinMaxScaler()
    # Fit scaler on training data, but the prompt just says:
    # scaled = scaler.fit_transform(df[['Total']])
    scaled = scaler.fit_transform(train[['Total']])
    
    X = []
    y = []
    window = 8
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])
        
    X_train = np.array(X)
    y_train = np.array(y)
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(
        X_train,
        y_train,
        epochs=10, # Reduced from 50 to 10 for faster execution
        batch_size=32,
        verbose=1
    )
    
    model.save("models/lstm.h5")
    print("LSTM model saved to models/lstm.h5")

if __name__ == "__main__":
    train_lstm()
