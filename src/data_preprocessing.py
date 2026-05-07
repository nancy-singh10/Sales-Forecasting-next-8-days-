import pandas as pd

def load_and_preprocess_data(file_path="data/sales.csv"):
    # Load Dataset
    df = pd.read_csv(file_path)
    
    # Clean Columns
    df.columns = df.columns.str.strip()
    
    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    
    # Convert Sales Column
    df['Total'] = (
        df['Total']
        .astype(str)
        .str.replace(',', '')
        .astype(float)
    )
    
    # Sort Dataset
    df = df.sort_values(['State', 'Date'])
    
    # Handle Missing Dates
    all_states = []
    
    for state in df['State'].unique():
        state_df = df[df['State'] == state]
        state_df = state_df.set_index('Date')
        
        # Resample to weekly frequency
        state_df = state_df.asfreq('W')
        
        # Interpolate Missing Sales
        state_df['Total'] = state_df['Total'].interpolate()
        
        # Re-add State Column
        state_df['State'] = state
        all_states.append(state_df)
    
    # Combine All States
    df_cleaned = pd.concat(all_states)
    
    return df_cleaned

if __name__ == "__main__":
    df = load_and_preprocess_data()
    print("Preprocessing completed. Dataset shape:", df.shape)
