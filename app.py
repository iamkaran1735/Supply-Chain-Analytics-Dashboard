import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Supply Chain Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Supply Chain Analytics Dashboard")
st.markdown("---")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin1")
    return df

df = load_data()

# -----------------------------
# Data Preparation
# -----------------------------
df["Delivery Gap"] = (
    df["Days for shipping (real)"] -
    df["Days for shipment (scheduled)"]
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Filters")

shipping = st.sidebar.multiselect(
    "Shipping Mode",
    df["Shipping Mode"].unique(),
    default=df["Shipping Mode"].unique()
)

region = st.sidebar.multiselect(
    "Region",
    df["Order Region"].unique(),
    default=df["Order Region"].unique()
)

filtered_df = df[
    (df["Shipping Mode"].isin(shipping)) &
    (df["Order Region"].isin(region))
]

# -----------------------------
# KPI Cards
# -----------------------------
total_orders = len(filtered_df)

avg_gap = filtered_df["Delivery Gap"].mean()

late_orders = len(filtered_df[filtered_df["Delivery Gap"] > 0])

late_ratio = (late_orders / total_orders) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", f"{total_orders:,}")

col2.metric("Average Delivery Gap", f"{avg_gap:.2f} Days")

col3.metric("Late Orders", f"{late_orders:,}")

col4.metric("Late Delivery %", f"{late_ratio:.2f}%")

st.markdown("---")
st.subheader("📦 Shipping Mode Distribution")

fig = px.histogram(
    filtered_df,
    x="Shipping Mode",
    color="Shipping Mode",
    title="Shipping Mode Distribution"
)

st.plotly_chart(fig, width="stretch")

filtered_df["Delivery Status"] = filtered_df["Delivery Gap"].apply(
    lambda x: "Delayed" if x > 0 else ("Early" if x < 0 else "On Time")
)

status = filtered_df["Delivery Status"].value_counts().reset_index()
status.columns = ["Delivery Status", "Count"]

fig = px.pie(
    status,
    names="Delivery Status",
    values="Count",
    title="Delivery Status Distribution",
    hole=0.4
)

st.plotly_chart(fig, width="stretch")

gap = (
    filtered_df.groupby("Shipping Mode")["Delivery Gap"]
    .mean()
    .reset_index()
)

fig = px.bar(
    gap,
    x="Shipping Mode",
    y="Delivery Gap",
    color="Shipping Mode",
    title="Average Delivery Gap by Shipping Mode"
)
fig.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig, width="stretch")

region_gap = (
    filtered_df.groupby("Order Region")["Delivery Gap"]
    .mean()
    .reset_index()
    .sort_values("Delivery Gap", ascending=False)
)

fig = px.bar(
    region_gap,
    x="Order Region",
    y="Delivery Gap",
    color="Delivery Gap",
    title="Average Delivery Gap by Region"
)
fig.update_layout(xaxis_tickangle=-45)


st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.subheader("📋 Dataset Preview")

st.dataframe(filtered_df.head(100))


st.download_button(
    "📥 Download CSV",
    filtered_df.to_csv(index=False),
    file_name="Filtered_Data.csv",
    mime="text/csv"
)

