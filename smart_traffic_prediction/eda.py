import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from plotting_utils import (
    plot_missingness_bar,
    plot_univariate_bar,
    plot_histogram,
    plot_time_series,
    plot_heatmap,
    plot_crosstab_bar
)

def load_cleaned_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def missingness_analysis(df: pd.DataFrame):
    """
    Figure 1: Bar chart of top‐20 columns by missing count.
    """
    missing_counts = df.isnull().sum().sort_values(ascending=False).head(20)
    plt.figure()
    plot_missingness_bar(missing_counts, title="Figure 1: Top 20 Columns by Missing Count")
    plt.savefig('figures/Figure1_missingness.png', dpi=300)
    plt.close()

def univariate_distributions(df: pd.DataFrame):
    """
    Figures 2–5:
      2: Bar chart of accident counts by Severity.
      3: Horizontal bar chart of top 10 Weather_Condition categories.
      4: Histogram of Temperature(F).
      5: Histogram of Visibility(mi).
    """
    # Figure 2: Severity distribution
    if 'Severity' in df.columns:
        counts_sev = df['Severity'].value_counts().sort_index()
        plt.figure()
        plot_univariate_bar(
            counts_sev,
            xlabel="Severity Level", ylabel="Count",
            title="Figure 2: Accident Counts by Severity"
        )
        plt.savefig('figures/Figure2_severity.png', dpi=300)
        plt.close()

    # Figure 3: Top 10 Weather Conditions
    if 'Weather_Condition' in df.columns:
        top10 = df['Weather_Condition'].value_counts().head(10)
        plt.figure()
        plot_univariate_bar(
            top10,
            horizontal=True,
            xlabel="Count", ylabel="Weather Condition",
            title="Figure 3: Top 10 Weather Conditions"
        )
        plt.savefig('figures/Figure3_weather.png', dpi=300)
        plt.close()

    # Figure 4: Temperature distribution
    if 'Temperature(F)' in df.columns:
        plt.figure()
        plot_histogram(
            df['Temperature(F)'].dropna(),
            bins=50,
            xlabel="Temperature (°F)",
            ylabel="Frequency",
            title="Figure 4: Temperature Distribution"
        )
        plt.savefig('figures/Figure4_temperature.png', dpi=300)
        plt.close()

    # Figure 5: Visibility distribution
    if 'Visibility(mi)' in df.columns:
        plt.figure()
        plot_histogram(
            df['Visibility(mi)'].dropna(),
            bins=30,
            xlabel="Visibility (mi)",
            ylabel="Frequency",
            title="Figure 5: Visibility Distribution"
        )
        plt.savefig('figures/Figure5_visibility.png', dpi=300)
        plt.close()

def temporal_patterns(df: pd.DataFrame):
    """
    Figures 6–8:
      6: Bar chart of accident frequency by Hour.
      7: Bar chart of accident frequency by DayOfWeek.
      8: Line plot of monthly accident counts Jan 2016–Mar 2023.
    """
    # Figure 6: Hour‐of‐Day distribution
    counts_hour = df['Hour'].value_counts().sort_index()
    plt.figure()
    plot_univariate_bar(
        counts_hour,
        xlabel="Hour of Day", ylabel="Accident Count",
        title="Figure 6: Accident Frequency by Hour"
    )
    plt.savefig('figures/Figure6_hourly.png', dpi=300)
    plt.close()

    # Figure 7: DayOfWeek distribution
    counts_dow = df['DayOfWeek'].value_counts().sort_index()
    plt.figure()
    plot_univariate_bar(
        counts_dow,
        xlabel="Day of Week (0=Mon…6=Sun)",
        ylabel="Accident Count",
        title="Figure 7: Accident Count by DayOfWeek"
    )
    plt.savefig('figures/Figure7_dayofweek.png', dpi=300)
    plt.close()

    # Figure 8: Monthly trend
    df['YearMonth'] = df['Start_Time'].dt.to_period('M')
    monthly = df.groupby('YearMonth').size()
    plt.figure()
    plot_time_series(
        monthly.index.to_timestamp(),
        monthly,
        xlabel="Month", ylabel="Accident Count",
        title="Figure 8: Monthly Accident Counts (2016–2023)"
    )
    plt.savefig('figures/Figure8_monthly.png', dpi=300)
    plt.close()

def spatial_distribution(df: pd.DataFrame):
    """
    Figure 9: Density heatmap of Start_Lat vs. Start_Lng (scatter with alpha).
    """
    sample = df.sample(n=20000, random_state=42)
    plt.figure()
    plot_heatmap(
        sample['Start_Lat'], sample['Start_Lng'],
        xlabel="Latitude", ylabel="Longitude",
        title="Figure 9: Spatial Heatmap of Accident Locations"
    )
    plt.savefig('figures/Figure9_spatial.png', dpi=300)
    plt.close()

def weather_vs_severity_crosstab(df: pd.DataFrame):
    """
    Figure 10: Cross‐tab of (Weather group) vs Severity, shown as clustered bar chart.
    We'll bin weather into four broad groups: Clear/Fair, Rain, Snow, Other.
    """
    def weather_group(w):
        if pd.isnull(w):
            return 'Other'
        w = w.lower()
        if 'rain' in w:
            return 'Rain'
        elif 'snow' in w:
            return 'Snow'
        elif 'fair' in w or 'clear' in w:
            return 'Clear/Fair'
        else:
            return 'Other'

    df['Weather_Group'] = df['Weather_Condition'].apply(weather_group)
    plt.figure()
    plot_crosstab_bar(
        df['Weather_Group'], df['Severity'],
        xlabel="Weather Group", ylabel="Count",
        title="Figure 10: Severity Distribution by Weather Group"
    )
    plt.savefig('figures/Figure10_crosstab.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    # Ensure the `figures` directory exists
    os.makedirs('figures', exist_ok=True)

    df_clean = load_cleaned_data('cleaned_accidents.parquet')

    # Generate and save Figures 1–10
    missingness_analysis(df_clean)
    univariate_distributions(df_clean)
    temporal_patterns(df_clean)
    spatial_distribution(df_clean)
    weather_vs_severity_crosstab(df_clean)

    print("EDA figures saved under `figures/`.")
