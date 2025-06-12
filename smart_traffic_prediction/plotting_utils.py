# plotting_utils.py

import matplotlib.pyplot as plt
import numpy as np

def plot_missingness_bar(missing_counts, title="Missingness"):
    """
    missing_counts: pandas Series indexed by column name, with number of missing values.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(missing_counts.index, missing_counts.values, color='steelblue')
    plt.xlabel("Number of Missing Values")
    plt.title(title)
    plt.tight_layout()

def plot_univariate_bar(value_counts, xlabel="", ylabel="", title="", horizontal=False):
    """
    value_counts: pandas Series with index=category, values=count.
    """
    plt.figure(figsize=(8, 5))
    if horizontal:
        plt.barh(value_counts.index.astype(str), value_counts.values, color='teal')
    else:
        plt.bar(value_counts.index.astype(str), value_counts.values, color='teal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

def plot_histogram(series, bins=30, xlabel="", ylabel="Frequency", title=""):
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=bins, color='coral', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

def plot_time_series(dates, counts, xlabel="", ylabel="", title=""):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, counts, marker='o', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

def plot_heatmap(lats, longs, xlabel="", ylabel="", title="", bins=200):
    plt.figure(figsize=(8, 6))
    plt.hist2d(longitudes:=longs.values, latitudes:=lats.values,
               bins=bins, cmap='inferno', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(label="Log‐Count")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

def plot_crosstab_bar(col1, col2, xlabel="", ylabel="", title=""):
    """
    Plot a clustered bar chart for counts of col2 categories within each col1 category.
    """
    import pandas as pd
    ct = pd.crosstab(col1, col2)
    ct.plot(kind='bar', figsize=(8, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
