# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def encode_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical hour features using sin/cos of Hour.
    Also create sine/cos for day of week if desired.
    """
    df['Hour_rad'] = 2 * np.pi * df['Hour'] / 24
    df['Sin_Hour'] = np.sin(df['Hour_rad'])
    df['Cos_Hour'] = np.cos(df['Hour_rad'])

    # Encode DayOfWeek similarly
    df['Dow_rad'] = 2 * np.pi * df['DayOfWeek'] / 7
    df['Sin_Dow'] = np.sin(df['Dow_rad'])
    df['Cos_Dow'] = np.cos(df['Dow_rad'])
    return df

def flag_weather_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary flags for major weather categories:
      - Weather_Rain: 1 if 'rain' in description.
      - Weather_Snow: 1 if 'snow' in description.
      - Weather_Fog: 1 if 'fog' or 'low visibility' etc.
      - Weather_Other: 1 if none of the above and not 'clear/fair'.
    """
    df['Weather_Lower'] = df['Weather_Condition'].str.lower().fillna('')
    df['Weather_Rain'] = df['Weather_Lower'].str.contains('rain').astype(int)
    df['Weather_Snow'] = df['Weather_Lower'].str.contains('snow').astype(int)
    df['Weather_Fog'] = df['Weather_Lower'].str.contains('fog|mist|haze|low visibility').astype(int)
    df['Weather_Clear'] = df['Weather_Lower'].str.contains('clear|fair').astype(int)
    df['Weather_Other'] = (
        (~df[['Weather_Rain', 'Weather_Snow', 'Weather_Fog', 'Weather_Clear']].any(axis=1))
        .astype(int)
    )
    df = df.drop(columns=['Weather_Lower'])
    return df

def encode_spatial_dummies(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """
    One-hot encode a high-cardinality column (e.g. 'State' or 'City'). 
    We recommend using only top‐k categories and group the rest into "Other" to avoid explosion.
    """
    top_categories = df[column].value_counts().nlargest(20).index
    df[f'{column}_Reduced'] = df[column].where(df[column].isin(top_categories), other='Other')

    # NOTE: use sparse_output=False instead of deprecated sparse=False
    encoder = OneHotEncoder(sparse_output=False, drop='if_binary')
    arr = encoder.fit_transform(df[[f'{column}_Reduced']])
    ohe_cols = [f"{prefix}_{cat}" for cat in encoder.categories_[0]]
    df_ohe = pd.DataFrame(arr, columns=ohe_cols, index=df.index)

    df = pd.concat([df, df_ohe], axis=1)
    df = df.drop(columns=[f'{column}_Reduced'])
    return df

def prepare_modeling_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the feature engineering pipeline:
    - Calls temporal encoding
    - Calls weather flags
    - Spatial dummies for 'State' and 'City'
    - Selects final columns
    """
    df = encode_temporal_features(df)
    df = flag_weather_conditions(df)

    # Spatial features
    if 'State' in df.columns:
        df = encode_spatial_dummies(df, column='State', prefix='State')
    if 'City' in df.columns:
        df = encode_spatial_dummies(df, column='City', prefix='City')

    # Select numeric features + boolean flags
    feature_cols = [
        'Sin_Hour', 'Cos_Hour', 'Sin_Dow', 'Cos_Dow',
        'Weather_Rain', 'Weather_Snow', 'Weather_Fog', 'Weather_Other', 'Weather_Clear',
        'Is_Day'
    ]

    # Add any one‐hot columns that start with "State_" or "City_"
    feature_cols += [c for c in df.columns if c.startswith('State_') or c.startswith('City_')]

    # Add spatial coordinates
    feature_cols += ['Start_Lat', 'Start_Lng']

    # Some continuous weather variables
    feature_cols += ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']

    # Drop rows where Severity is missing (target variable)
    df = df[df['Severity'].notnull()]

    # Now drop any rows that still have NaN in any of our chosen features
    final_cols = feature_cols + ['Severity']
    df = df.dropna(subset=final_cols)

    return df[final_cols]

if __name__ == '__main__':
    cleaned = pd.read_parquet('cleaned_accidents.parquet')
    model_df = prepare_modeling_dataframe(cleaned)
    model_df.to_parquet('model_ready_data.parquet', index=False)
    print("Saved feature‐engineered data to 'model_ready_data.parquet'")
