# Delivery Performance Analysis Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin1")
# Display Dataset
print("First 5 Rows")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================
# Data Cleaning

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# Missing Values Percentage
missing = (df.isnull().sum() / len(df)) * 100

print("\nMissing Values (%)")
print(missing.sort_values(ascending=False).head(15))

# =====================================
# Basic Statistics
# =====================================

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# =====================================
# Exploratory Data Analysis (EDA)

import os

# Create images folder if not exists
os.makedirs("images", exist_ok=True)

# Shipping Mode Count
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Shipping Mode")
plt.title("Shipping Mode Distribution")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("images/shipping_mode.png")
plt.show()

# -------------------------------
# Delivery Gap Calculation
# -------------------------------

df["Delivery Gap"] = (
    df["Days for shipping (real)"] -
    df["Days for shipment (scheduled)"]
)

print("\nDelivery Gap (First 5 Rows)")
print(df[[
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Gap"
]].head())
print("Program Finished Successfully")

# -------------------------------
# Delivery Status Classification
# -------------------------------

def classify_delivery(gap):
    if gap < 0:
        return "Early"
    elif gap == 0:
        return "On Time"
    else:
        return "Delayed"

df["Delivery Status"] = df["Delivery Gap"].apply(classify_delivery)

print("\nDelivery Status Count")
print(df["Delivery Status"].value_counts())

# Delivery Status Distribution

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Delivery Status",
    order=["Early","On Time","Delayed"]
)

plt.title("Delivery Status Distribution")
plt.xlabel("Delivery Status")
plt.ylabel("Count")

plt.show()

plt.close()

# -------------------------------
# KPI 1 : On-Time Delivery Rate
# -------------------------------

total_orders = len(df)
on_time_orders = len(df[df["Delivery Status"] == "On Time"])

on_time_rate = (on_time_orders / total_orders) * 100

print("\n========== KPI ==========")
print(f"Total Orders : {total_orders}")
print(f"On Time Orders : {on_time_orders}")
print(f"On Time Delivery Rate : {on_time_rate:.2f}%")

# -------------------------------
# KPI 2 : Average Delivery Delay
# -------------------------------

avg_delay = df["Delivery Gap"].mean()

print(f"Average Delivery Delay : {avg_delay:.2f} Days")

# -------------------------------
# KPI 3 : Late Delivery Risk Ratio
# -------------------------------

late_orders = len(df[df["Delivery Gap"] > 0])

late_ratio = (late_orders / total_orders) * 100

print(f"Late Delivery Risk Ratio : {late_ratio:.2f}%")

# ---------------------------------------
# Shipping Mode Efficiency Analysis
# ---------------------------------------

shipping_analysis = df.groupby("Shipping Mode")["Delivery Gap"].mean().sort_values()

print("\nAverage Delivery Gap by Shipping Mode")
print(shipping_analysis)

plt.figure(figsize=(8,5))

shipping_analysis.plot(
    kind="bar",
    color=["green","blue","orange","red"]
)

plt.title("Average Delivery Gap by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Average Delivery Gap (Days)")
plt.xticks(rotation=20)

plt.show()

# ---------------------------------------
# Region Wise Delay Analysis
# ---------------------------------------

region_delay = df.groupby("Order Region")["Delivery Gap"].mean().sort_values(ascending=False)

print("\nAverage Delivery Gap by Region")
print(region_delay)

plt.figure(figsize=(10,6))

region_delay.plot(kind="bar")

plt.title("Average Delivery Gap by Region")
plt.xlabel("Region")
plt.ylabel("Average Delivery Gap (Days)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ---------------------------------------
# Market Wise Delay Analysis
# ---------------------------------------

market_delay = df.groupby("Market")["Delivery Gap"].mean().sort_values(ascending=False)

print("\nAverage Delivery Gap by Market")
print(market_delay)

plt.figure(figsize=(10,5))

market_delay.plot(kind="bar")

plt.title("Average Delivery Gap by Market")
plt.xlabel("Market")
plt.ylabel("Average Delivery Gap (Days)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ---------------------------------------
# Customer Segment Analysis
# ---------------------------------------

segment_delay = df.groupby("Customer Segment")["Delivery Gap"].mean().sort_values(ascending=False)

print("\nAverage Delivery Gap by Customer Segment")
print(segment_delay)

plt.figure(figsize=(8,5))

segment_delay.plot(kind="bar")

plt.title("Average Delivery Gap by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Delivery Gap (Days)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# ---------------------------------------
# Heatmap: Region vs Shipping Mode
# ---------------------------------------

import seaborn as sns

heatmap_data = pd.pivot_table(
    df,
    values="Delivery Gap",
    index="Order Region",
    columns="Shipping Mode",
    aggfunc="mean"
)

print("\nHeatmap Data")
print(heatmap_data)

plt.figure(figsize=(12,8))

sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="YlOrRd",
    fmt=".2f"
)

plt.title("Average Delivery Gap by Region and Shipping Mode")

plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include=['number'])

corr = numeric_df.corr()

print("\nCorrelation Matrix")
print(corr)

plt.figure(figsize=(14,10))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ---------------------------------------
# Correlation Heatmap
# ---------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt

numeric_df = df.select_dtypes(include=['number'])

corr = numeric_df.corr()

print("\nCorrelation Matrix")
print(corr)

plt.figure(figsize=(14,10))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()