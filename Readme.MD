# Dashboard Analisis Data Penjualan E-Commerce

## Deskripsi Singkat
Project ini merupakan dashboard interaktif yang digunakan untuk menganalisis data penjualan, performa produk, dan perilaku pelanggan pada platform e-commerce. Dengan dashboard ini, pengguna dapat memahami tren bisnis, mengeksplorasi kategori produk terlaris, hingga menganalisis segmen pelanggan berbasis RFM.

---

## Daftar File Proyek

- `Project_analisis_data.ipynb` — Notebook utama untuk eksplorasi dan analisis data.
- `dashboard.py` — Script Python untuk Streamlit dashboard (jika tersedia).
- `requirements.txt` — Daftar dependensi Python.
- Berbagai file dataset, misal:
  - `orders_dataset.csv`
  - `order_items_dataset.csv`
  - `order_payments_dataset.csv`
  - `order_reviews_dataset.csv`
  - `customers_dataset.csv`
  - `products_dataset.csv`
  - `product_category_name_translation.csv`
  - `sellers_dataset.csv`

---

## Cara Menjalankan Project

### **1. Jalankan di Jupyter Notebook**
- Buka `Project_analisis_data.ipynb` menggunakan Jupyter Notebook/JupyterLab.
- Jalankan sel-sel kode untuk melakukan analisis data, visualisasi, dan eksplorasi insight.

### **2. Jalankan Dashboard Interaktif (jika ada Streamlit)**
1. Pastikan semua file dataset berada di satu folder dengan `dashboard.py`.
2. Install semua dependensi:
    ```bash
    pip install -r requirements.txt
    ```
3. Jalankan dashboard dengan perintah:
    ```bash
    streamlit run dashboard.py
    ```
4. Dashboard akan terbuka otomatis di browser Anda.

---

## Dependensi

Semua dependensi Python tercantum pada `requirements.txt`.  
Beberapa library utama:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `streamlit`
- `babel`

---

## Ringkasan Analisis & Insight

- **Tren Penjualan Harian:** Visualisasi jumlah order & revenue per hari untuk melihat seasonality dan performa bisnis.
- **Performa Produk:** Analisis kategori produk dengan jumlah pesanan terbanyak dan paling sedikit.
- **Demografi Pelanggan:** Sebaran jumlah customer berdasarkan wilayah/state.
- **Analisis RFM:** Segmentasi pelanggan berdasarkan Recency, Frequency, dan Monetary untuk mengidentifikasi customer paling bernilai.

> Catatan: Insight lengkap dapat dilihat pada hasil visualisasi dan analisis di notebook.

---

## Kontributor

- **Nama:** Yohannes YudhaYudha
- **Email:** Yohannes.Haryo@lintasarta.co.id
- **ID Dicoding:** YH Yudha

---

## Lisensi
Project ini hanya digunakan untuk keperluan pembelajaran dan analisis data.

---
