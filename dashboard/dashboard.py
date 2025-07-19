

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from babel.numbers import format_currency

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Path folder data di Google Drive Anda
data_dir = '/content/drive/MyDrive/Colab Notebooks/Tugas YH Yudha - Proyek Analisis Data/data/'

# Daftar nama file csv yang ingin dibaca
csv_files = [
    'order_payments_dataset.csv',
    'order_items_dataset.csv',
    'orders_dataset.csv',
    'order_reviews_dataset.csv',
    'sellers_dataset.csv',
    'customers_dataset.csv',
    'product_category_name_translation.csv',
    'products_dataset.csv'
]

# Membaca semua file dan simpan ke dictionary
import pandas as pd
dfs = {}

for file in csv_files:
    file_path = data_dir + file
    try:
        df = pd.read_csv(file_path)
        dfs[file] = df
        print(f"✅ Berhasil memuat {file} dari path: {file_path}")
    except FileNotFoundError:
        print(f"❌ Error: File {file} tidak ditemukan di path: {file_path}")
    except Exception as e:
        print(f"❌ Terjadi error lain saat memuat {file}: {e}")

order_payments_dataset_df = dfs['order_payments_dataset.csv']
order_items_dataset_df = dfs['order_items_dataset.csv']
orders_dataset_df = dfs['orders_dataset.csv']
order_reviews_dataset_df = dfs['order_reviews_dataset.csv']
sellers_dataset_df = dfs['sellers_dataset.csv']
customers_dataset_df = dfs['customers_dataset.csv']
product_category_name_translation_df = dfs['product_category_name_translation.csv']
products_dataset_df = dfs['products_dataset.csv']

st.title("Visualization & Explanatory Analysis")

# --- Preprocessing ---
cols = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]
for col in cols:
    orders_dataset_df[col] = orders_dataset_df[col].fillna(pd.NaT)

datetime_columns = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date"
]
for column in datetime_columns:
    orders_dataset_df[column] = pd.to_datetime(orders_dataset_df[column])

# ----- Distribusi Status Order -----
st.subheader("Distribusi Status Order")
fig1, ax1 = plt.subplots(figsize=(8, 4))
orders_dataset_df['order_status'].value_counts().plot(kind='bar', color='skyblue', ax=ax1)
ax1.set_title('Distribusi Status Order')
ax1.set_xlabel('Status')
ax1.set_ylabel('Jumlah')
st.pyplot(fig1)
st.markdown("> **Insight:** Status 'delivered' mendominasi transaksi, diikuti status lain seperti canceled dan unavailable.")

# ----- Distribusi Kategori Produk -----
st.subheader("10 Kategori Produk Terpopuler")
order_product_df = order_items_dataset_df.merge(
    orders_dataset_df, on="order_id"
).merge(
    products_dataset_df, on="product_id"
).merge(
    product_category_name_translation_df, on="product_category_name", how="left"
)
fig2, ax2 = plt.subplots(figsize=(12, 5))
order_product_df['product_category_name_english'].value_counts().head(10).plot(kind='bar', color='orange', ax=ax2)
ax2.set_title('10 Kategori Produk Terpopuler')
ax2.set_xlabel('Kategori')
ax2.set_ylabel('Jumlah Order')
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# ----- Tren Order Harian -----
st.subheader("Tren Order Harian")
daily_orders = orders_dataset_df.groupby(orders_dataset_df['order_purchase_timestamp'].dt.date)['order_id'].nunique()
fig3, ax3 = plt.subplots(figsize=(14, 5))
daily_orders.plot(ax=ax3)
ax3.set_title('Tren Order Harian')
ax3.set_xlabel('Tanggal')
ax3.set_ylabel('Jumlah Order')
st.pyplot(fig3)

# ----- Tren Order Bulanan -----
st.subheader("Tren Order Bulanan")
orders_dataset_df['month'] = orders_dataset_df['order_purchase_timestamp'].dt.to_period('M')
monthly_orders = orders_dataset_df.groupby('month')['order_id'].nunique()
fig4, ax4 = plt.subplots(figsize=(12, 4))
monthly_orders.plot(marker='o', ax=ax4)
ax4.set_title('Tren Order Bulanan')
ax4.set_xlabel('Bulan')
ax4.set_ylabel('Jumlah Order')
st.pyplot(fig4)

# ----- Korelasi Jumlah Produk per Order VS Nilai Pembayaran -----
st.subheader("Korelasi Jumlah Produk per Order vs Total Pembayaran")
order_total = order_items_dataset_df.groupby('order_id').agg({
    'product_id':'count'
}).rename(columns={'product_id':'num_products'}).reset_index()
order_total = order_total.merge(
    order_payments_dataset_df.groupby('order_id')['payment_value'].sum().reset_index(),
    on='order_id', how='left'
)
fig5, ax5 = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=order_total, x='num_products', y='payment_value', alpha=0.4, ax=ax5)
ax5.set_title('Korelasi Jumlah Produk per Order vs Total Pembayaran')
ax5.set_xlabel('Jumlah Produk per Order')
ax5.set_ylabel('Total Pembayaran')
st.pyplot(fig5)

# ----- Distribusi Waktu Pengiriman -----
st.subheader("Distribusi Lama Pengiriman ke Pelanggan (hari)")
orders_dataset_df['delivery_days'] = (
    orders_dataset_df['order_delivered_customer_date'] - orders_dataset_df['order_purchase_timestamp']
).dt.days
fig6, ax6 = plt.subplots(figsize=(8, 5))
orders_dataset_df['delivery_days'].dropna().hist(bins=30, ax=ax6)
ax6.set_title('Distribusi Lama Pengiriman ke Pelanggan (hari)')
ax6.set_xlabel('Hari')
ax6.set_ylabel('Jumlah Order')
st.pyplot(fig6)

# ------------- EXPLANATORY ANALYSIS: PRODUCT PERFORMANCE -------------
st.header("Product Performance")
def create_sum_order_items_df(df):
    sum_order_items_df = (
        df.groupby("product_category_name_english")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="total_orders")
    )
    return sum_order_items_df

sum_order_items_df = create_sum_order_items_df(order_product_df)

fig7, ax7 = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), dpi=100)
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
# Best
sns.barplot(
    x="total_orders",
    y="product_category_name_english",
    data=sum_order_items_df.head(5),
    palette=colors,
    ax=ax7[0],
    legend=False
)
ax7[0].set_title("Best Performing Product", loc="center", fontsize=14)
# Worst
sns.barplot(
    x="total_orders",
    y="product_category_name_english",
    data=sum_order_items_df.sort_values(by="total_orders", ascending=True).head(5),
    palette=colors,
    ax=ax7[1],
    legend=False
)
ax7[1].set_title("Worst Performing Product", loc="center", fontsize=14)
st.pyplot(fig7)

# ------------- EXPLANATORY ANALYSIS: DEMOGRAPHY -------------
st.header("Customer Demography by State")
order_customer_df = order_items_dataset_df.merge(
    orders_dataset_df, on="order_id"
).merge(
    customers_dataset_df, on="customer_id"
)
def create_bystate_df(df):
    bystate_df = df.groupby(by="customer_state").customer_id.nunique().reset_index()
    bystate_df.rename(columns={"customer_id": "customer_count"}, inplace=True)
    return bystate_df
bystate_df = create_bystate_df(order_customer_df)
sorted_df = bystate_df.sort_values(by="customer_count", ascending=False)
fig8, ax8 = plt.subplots(figsize=(20, 10))
sns.barplot(
    x="customer_count",
    y="customer_state",
    data=sorted_df,
    palette=["#90CAF9" if i==0 else "#D3D3D3" for i in range(len(sorted_df))],
    ax=ax8
)
ax8.set_title("Number of Customers by State", loc="center", fontsize=30)
st.pyplot(fig8)

# ------------- DAILY ORDER METRICS -------------
st.header("Daily Order Metrics")
min_date = orders_dataset_df["order_purchase_timestamp"].min()
max_date = orders_dataset_df["order_purchase_timestamp"].max()

with st.sidebar:
    start_date, end_date = st.date_input(
        "Pilih Rentang Tanggal",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
orders_filtered = orders_dataset_df[
    (orders_dataset_df["order_purchase_timestamp"].dt.date >= start_date) &
    (orders_dataset_df["order_purchase_timestamp"].dt.date <= end_date)
]
order_payments_df = pd.merge(orders_filtered, order_payments_dataset_df, on="order_id", how="inner")
order_payments_df["total_price"] = order_payments_df["payment_value"]

def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "total_price": "sum"
    }).reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "total_price": "revenue"
    }, inplace=True)
    return daily_orders_df

daily_orders_df = create_daily_orders_df(order_payments_df)
total_orders = daily_orders_df["order_count"].sum()
total_revenue = daily_orders_df["revenue"].sum()
formatted_revenue = format_currency(total_revenue, "AUD", locale='es_CO')

col1, col2 = st.columns(2)
col1.metric("Total Orders", total_orders)
col2.metric("Total Revenue", formatted_revenue)

fig9, ax9 = plt.subplots(figsize=(16, 8))
ax9.plot(
    daily_orders_df["order_purchase_timestamp"],  # X-axis
    daily_orders_df["order_count"],              # Y-axis
    marker='o',
    linewidth=2
)
ax9.set_title("Daily Orders Over Time")
ax9.set_xlabel("Tanggal")
ax9.set_ylabel("Jumlah Order")
st.pyplot(fig9)

# ------------- RFM ANALYSIS -------------
st.header("Customer RFM Analysis")
def create_rfm_df(df):
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    recent_date = df["order_purchase_timestamp"].max().date()
    rfm_df = df.groupby("customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "price": "sum"
    })
    rfm_df.columns = ["customer_id", "last_purchase_date", "frequency", "monetary"]
    rfm_df["last_purchase_date"] = rfm_df["last_purchase_date"].dt.date
    rfm_df["recency"] = rfm_df["last_purchase_date"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("last_purchase_date", axis=1, inplace=True)
    return rfm_df

customer_order_df = customers_dataset_df.merge(
    orders_dataset_df, on="customer_id"
).merge(
    order_items_dataset_df, on="order_id", how="left"
)

rfm_df = create_rfm_df(customer_order_df)
avg_recency = round(rfm_df.recency.mean(), 1)
avg_frequency = round(rfm_df.frequency.mean(), 2)
avg_monetary = format_currency(rfm_df.monetary.mean(), "USD", locale='id_ID')
col3, col4, col5 = st.columns(3)
col3.metric("Average Recency (days)", avg_recency)
col4.metric("Average Frequency", avg_frequency)
col5.metric("Average Monetary", avg_monetary)

fig10, ax10 = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
colors = ["#90CAF9"] * 5

# By Recency (paling lama tidak belanja)
top_recency = rfm_df.sort_values(by="recency", ascending=False).head(5)
sns.barplot(
    y="recency",
    x="customer_id",
    hue="customer_id",
    data=top_recency,
    palette=colors,
    ax=ax10[0],
    legend=False
)
ax10[0].set_title("By Recency (days)\n(Paling Lama Tidak Belanja)", loc="center", fontsize=40)
ax10[0].set_xlabel("customer_id", fontsize=25)
ax10[0].tick_params(axis='x', labelsize=24, labelrotation=85)
ax10[0].tick_params(axis='y', labelsize=22)

# By Frequency
top_frequency = rfm_df.sort_values(by="frequency", ascending=False).head(5)
sns.barplot(
    y="frequency",
    x="customer_id",
    hue="customer_id",
    data=top_frequency,
    palette=colors,
    ax=ax10[1],
    legend=False
)
ax10[1].set_title("By Frequency", loc="center", fontsize=40)
ax10[1].set_xlabel("customer_id", fontsize=25)
ax10[1].tick_params(axis='x', labelsize=24, labelrotation=85)
ax10[1].tick_params(axis='y', labelsize=22)

# By Monetary
top_monetary = rfm_df.sort_values(by="monetary", ascending=False).head(5)
sns.barplot(
    y="monetary",
    x="customer_id",
    hue="customer_id",
    data=top_monetary,
    palette=colors,
    ax=ax10[2],
    legend=False
)
ax10[2].set_title("By Monetary", loc="center", fontsize=40)
ax10[2].set_xlabel("customer_id", fontsize=25)
ax10[2].tick_params(axis='x', labelsize=24, labelrotation=85)
ax10[2].tick_params(axis='y', labelsize=22)

st.pyplot(fig10)

st.caption('Copyright © Laskar AI 2025')
