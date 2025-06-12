#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda_us_accidents.py

Basic Exploratory Data Analysis (EDA) for the US-Accidents sampled dataset (500 K rows).
Assumes the file 'US_Accidents_sample.csv' is in the same folder.

Usage:
    python eda_us_accidents.py

Dependencies:
    - pandas
    - numpy
    - matplotlib
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1) CONFIGURATION: filepaths and figure‐output directory
# -----------------------------------------------------------------------------

DATA_FILE = "US_Accidents_sample.csv"
FIG_DIR = "figures"

# Create the figures directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 2) LOAD DATA (with explicit datetime conversion)
# -----------------------------------------------------------------------------

print(f"Loading data from '{DATA_FILE}'...")
df = pd.read_csv(DATA_FILE)

# Force-convert Start_Time, End_Time, Weather_Timestamp to datetime:
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
df["Weather_Timestamp"] = pd.to_datetime(df["Weather_Timestamp"], errors="coerce")

print("Dataset shape:", df.shape)
print()

# -----------------------------------------------------------------------------
# 3) QUICK OVERVIEW (INFO, DESCRIBE, MISSINGNESS)
# -----------------------------------------------------------------------------

print("1) DataFrame.info():")
print(df.info())
print()

print("2) DataFrame.describe() [numeric columns]:")
print(df.describe(include=[np.number]).T)
print()

# Check missing‐value counts for key columns
missing_counts = df.isna().sum().sort_values(ascending=False)
print("3) Missing‐value counts (top 20 columns):")
print(missing_counts.head(20))
print()

# Save missingness table to CSV (optional)
missing_counts.to_csv(os.path.join(FIG_DIR, "missing_counts.csv"), header=["num_missing"])


# -----------------------------------------------------------------------------
# 4) UNIVARIATE ANALYSES: DISTRIBUTIONS & BAR CHARTS
# -----------------------------------------------------------------------------

# 4.1) Severity distribution (1–4)
plt.figure(figsize=(6, 4))
severity_counts = df["Severity"].value_counts().sort_index()
plt.bar(severity_counts.index.astype(int), severity_counts.values, width=0.6, color="skyblue", edgecolor="k")
plt.xticks([1, 2, 3, 4])
plt.xlabel("Severity (1=Low impact → 4=High impact)")
plt.ylabel("Number of accidents")
plt.title("Accident Severity Distribution")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "severity_distribution.png"), dpi=150)
plt.close()

# 4.2) Top 10 Weather_Condition categories (bar chart)
plt.figure(figsize=(8, 5))
top_weather = df["Weather_Condition"].value_counts().head(10)
plt.barh(top_weather.index[::-1], top_weather.values[::-1], color="salmon", edgecolor="k")
plt.xlabel("Count")
plt.title("Top 10 Weather Conditions at Time of Accident")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "top10_weather_conditions.png"), dpi=150)
plt.close()

# 4.3) Histogram of Temperature (°F)
plt.figure(figsize=(6, 4))
temps = df["Temperature(F)"].dropna()
plt.hist(temps, bins=50, edgecolor="k", alpha=0.7)
plt.xlabel("Temperature (°F)")
plt.ylabel("Frequency")
plt.title("Distribution of Temperature at Accident Time")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "temperature_histogram.png"), dpi=150)
plt.close()

# 4.4) Histogram of Visibility (miles)
plt.figure(figsize=(6, 4))
vis = df["Visibility(mi)"].dropna()
plt.hist(vis, bins=40, edgecolor="k", alpha=0.7)
plt.xlabel("Visibility (mi)")
plt.ylabel("Frequency")
plt.title("Distribution of Visibility at Accident Time")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "visibility_histogram.png"), dpi=150)
plt.close()

# 4.5) Histogram of Wind_Speed (mph)
plt.figure(figsize=(6, 4))
ws = df["Wind_Speed(mph)"].dropna()
plt.hist(ws, bins=40, edgecolor="k", alpha=0.7)
plt.xlabel("Wind Speed (mph)")
plt.ylabel("Frequency")
plt.title("Distribution of Wind Speed at Accident Time")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "windspeed_histogram.png"), dpi=150)
plt.close()

# 4.6) Bar chart of top 10 States (by accident count)
plt.figure(figsize=(8, 5))
top_states = df["State"].value_counts().head(10)
plt.barh(top_states.index[::-1], top_states.values[::-1], color="lightgreen", edgecolor="k")
plt.xlabel("Count")
plt.title("Top 10 States by Number of Accidents (Sampled Data)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "top10_states.png"), dpi=150)
plt.close()


# -----------------------------------------------------------------------------
# 5) TIME SERIES ANALYSIS: ACCIDENT COUNTS OVER TIME
# -----------------------------------------------------------------------------

# 5.1) Create Year‐Month column from Start_Time
# Ensure Start_Time is datetime; if conversion failed, filter out NaT
df = df[df["Start_Time"].notna()]
df["YearMonth"] = df["Start_Time"].dt.to_period("M")
acc_per_month = df.groupby("YearMonth").size()

# Convert PeriodIndex to datetime at month start for plotting
acc_per_month.index = acc_per_month.index.to_timestamp()

plt.figure(figsize=(10, 4))
plt.plot(acc_per_month.index, acc_per_month.values, color="navy", linewidth=1)
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.title("Accidents per Month (Jan 2016 – Mar 2023, Sampled)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "accidents_per_month.png"), dpi=150)
plt.close()

# 5.2) Total accidents per year
acc_per_year = df["Start_Time"].dt.year.value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(acc_per_year.index.astype(int), acc_per_year.values, color="orchid", edgecolor="k")
plt.xlabel("Year")
plt.ylabel("Total Accidents")
plt.title("Total Accidents per Year (Sampled)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "accidents_per_year.png"), dpi=150)
plt.close()


# -----------------------------------------------------------------------------
# 6) TEMPORAL PATTERNS
# -----------------------------------------------------------------------------

# 6.1) Extract HourOfDay and DayOfWeek
df["HourOfDay"] = df["Start_Time"].dt.hour
df["DayOfWeek"] = df["Start_Time"].dt.day_name()

# 6.2) Figure 6: Bar chart of accident frequency by hour of day
plt.figure(figsize=(8, 5))
hour_counts = df["HourOfDay"].value_counts().sort_index()
plt.bar(hour_counts.index, hour_counts.values, color="steelblue", edgecolor="k")
plt.xticks(range(0, 24))
plt.xlabel("Hour of Day (0–23)")
plt.ylabel("Number of Accidents")
plt.title("Accident Frequency by Hour of Day")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "accidents_by_hour.png"), dpi=150)
plt.close()

# 6.3) Figure 7: Bar chart of accident counts by day of week
plt.figure(figsize=(8, 5))
# Ensure weekdays are in order Monday → Sunday
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_counts = df["DayOfWeek"].value_counts().reindex(weekday_order)
plt.bar(day_counts.index, day_counts.values, color="coral", edgecolor="k")
plt.xlabel("Day of Week")
plt.ylabel("Number of Accidents")
plt.title("Accident Counts by Day of Week")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "accidents_by_dayofweek.png"), dpi=150)
plt.close()

# 6.4) Figure 8 (re‐plotted here for clarity): Line plot of monthly accident counts
plt.figure(figsize=(10, 4))
plt.plot(acc_per_month.index, acc_per_month.values, color="navy", linewidth=1)
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.title("Accidents per Month (Jan 2016 – Mar 2023, Sampled)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "monthly_trends_lineplot.png"), dpi=150)
plt.close()


# -----------------------------------------------------------------------------
# 7) SPATIAL DISTRIBUTION
# -----------------------------------------------------------------------------

# 7.1) Placeholder Figure 9: State‐level accident counts (bar chart as proxy for choropleth)
# Note: For a true choropleth you'd need a shapefile and geopandas. Here we plot counts per state.
plt.figure(figsize=(10, 6))
state_counts = df["State"].value_counts().sort_values(ascending=False)
plt.barh(state_counts.index[::-1], state_counts.values[::-1], color="mediumseagreen", edgecolor="k")
plt.xlabel("Number of Accidents")
plt.title("Accident Counts by State (Sampled) – Proxy for Choropleth")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "accidents_by_state_bar.png"), dpi=150)
plt.close()

# 7.2) Figure 10: Point‐density heatmap of Start_Lat vs. Start_Lng
plt.figure(figsize=(8, 6))
lat = df["Start_Lat"].dropna()
lng = df["Start_Lng"].dropna()
plt.hist2d(lng, lat, bins=300, cmap="hot", norm=plt.matplotlib.colors.LogNorm())
plt.colorbar(label="Log(Number of Accidents)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Heatmap of Accident Locations (Sampled)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "spatial_heatmap.png"), dpi=150)
plt.close()


# -----------------------------------------------------------------------------
# 8) WEATHER VS. SEVERITY CROSS‐TABULATION
# -----------------------------------------------------------------------------

# 8.1) Categorize Weather_Condition into broad classes
def categorize_weather(cond):
    if pd.isna(cond):
        return "Other"
    cond_lower = cond.lower()
    if "snow" in cond_lower:
        return "Snow"
    elif ("rain" in cond_lower) or ("drizzle" in cond_lower) or ("thunderstorm" in cond_lower):
        return "Rain"
    elif ("clear" in cond_lower) or ("fair" in cond_lower) or ("sunny" in cond_lower):
        return "Clear/Fair"
    else:
        return "Other"

df["Weather_Class"] = df["Weather_Condition"].map(categorize_weather)

# 8.2) Figure 11: Clustered bar chart comparing severity distributions across weather categories
weather_sev = df.groupby(["Weather_Class", "Severity"]).size().unstack(fill_value=0)
weather_sev_pct = weather_sev.div(weather_sev.sum(axis=1), axis=0) * 100

# Plot absolute counts (stacked bar for clarity)
plt.figure(figsize=(8, 6))
weather_sev.plot(
    kind="bar", 
    stacked=False, 
    color=["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"], 
    edgecolor="k"
)
plt.xlabel("Weather Category")
plt.ylabel("Number of Accidents")
plt.title("Accident Severity by Weather Category (Sampled)")
plt.legend(title="Severity", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "severity_by_weather_counts.png"), dpi=150)
plt.close()

# Plot percentage distribution (clustered)
plt.figure(figsize=(8, 6))
weather_sev_pct.plot(
    kind="bar",
    stacked=False,
    color=["#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00"],
    edgecolor="k"
)
plt.xlabel("Weather Category")
plt.ylabel("Percentage of Accidents (%)")
plt.title("Accident Severity Distribution (%) by Weather Category")
plt.legend(title="Severity", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "severity_by_weather_pct.png"), dpi=150)
plt.close()


# -----------------------------------------------------------------------------
# 9) SAVE A SMALL SUMMARY CSV (OPTIONAL)
# -----------------------------------------------------------------------------

# Example: Save the accidents‐per‐month data for further analysis
acc_per_month_df = pd.DataFrame({
    "YearMonth": acc_per_month.index.strftime("%Y-%m"),
    "AccidentCount": acc_per_month.values
})
acc_per_month_df.to_csv(os.path.join(FIG_DIR, "accidents_per_month.csv"), index=False)

# Save weather vs severity counts for reporting
weather_sev.to_csv(os.path.join(FIG_DIR, "weather_vs_severity_counts.csv"))
weather_sev_pct.to_csv(os.path.join(FIG_DIR, "weather_vs_severity_pct.csv"))

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------

print(f"EDA complete. All figures saved to the '{FIG_DIR}' folder.")
