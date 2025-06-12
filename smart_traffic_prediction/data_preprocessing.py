# data_preprocessing.py

import pandas as pd
import numpy as np

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the 500k‐row sample CSV into a DataFrame, parsing date columns if possible.
    """
    df = pd.read_csv(path,
                     parse_dates=['Start_Time', 'End_Time', 'Weather_Timestamp'],
                     infer_datetime_format=True)
    return df

def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop columns with >60% missingness if deemed unusable.
    - Impute remaining weather and coordinate values.
    - Cast boolean flags to int.
    - Drop duplicate IDs.
    """
    # 1) Drop columns that are >60% missing
    missing_frac = df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > 0.60].index.tolist()
    df = df.drop(columns=cols_to_drop)

    # 2) Impute remaining numerical weather fields with median
    weather_cols = [
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
        'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
        'Precipitation(in)'
    ]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 3) Impute or fill missing coordinates
    if 'Start_Lat' in df.columns and 'Start_Lng' in df.columns:
        mask_missing = df['Start_Lat'].isnull() | df['Start_Lng'].isnull()
        if mask_missing.any():
            coords_median = df[['Start_Lat', 'Start_Lng']].median()
            df.loc[mask_missing, ['Start_Lat', 'Start_Lng']] = coords_median

    # 4) Cast boolean flags to integer (0/1)
    bool_cols = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
        'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
        'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 5) Drop duplicate IDs
    df = df.drop_duplicates(subset=['ID'])

    return df

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From Start_Time, extract:
      - Year, Month, DayOfWeek, Hour
      - A boolean flag for Day/Night via Sunrise_Sunset
    """
    # Ensure Start_Time is datetime
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

    # Now it's safe to use .dt accessor
    df['Year'] = df['Start_Time'].dt.year
    df['Month'] = df['Start_Time'].dt.month
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek  # Monday=0…Sunday=6
    df['Hour'] = df['Start_Time'].dt.hour

    # Encode Sunrise_Sunset: “Day”=1, “Night”=0
    if 'Sunrise_Sunset' in df.columns:
        df['Is_Day'] = (df['Sunrise_Sunset'] == 'Day').astype(int)
    else:
        df['Is_Day'] = np.nan

    return df

if __name__ == '__main__':
    raw = load_raw_data('data/US_Accidents_sample.csv')
    cleaned = clean_and_filter(raw)
    enriched = add_datetime_features(cleaned)
    enriched.to_parquet('cleaned_accidents.parquet', index=False)
    print("Saved cleaned data to 'cleaned_accidents.parquet'")
