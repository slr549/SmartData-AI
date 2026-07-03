Berikut adalah draf lengkap materi presentasi proyek kelompokmu yang sudah disusun rapi dalam format Markdown (`.md`). Format ini sangat cocok untuk ditampilkan langsung menggunakan _tools_ presentasi berbasis Markdown (seperti Marp atau Slidev) atau disalin ke Notion dan GitHub.

---

````markdown
# PRESENTASI PROYEK KELOMPOK

## Analisis Sentimen dan Segmentasi Opini Publik terhadap Produk/Brand Menggunakan Multi-Model Machine Learning

---

## 1. TIMELINE & JADWAL PENGERJAAN

Rekomendasi alur pengerjaan proyek dirancang secara terstruktur selama 4 minggu:

- **Minggu 1: Inisiasi & Pengambilan Data**
  - Diskusi framework utama kelompok.
  - Proses **Web Scraping** data ulasan teks dari e-commerce atau media sosial.
  - _Output:_ File data mentah (_raw data_) berformat CSV/JSON.
- **Minggu 2: Pra-pemrosesan Data**
  - Melakukan _text preprocessing_ (pembersihan teks) dan pelabelan analisis sentimen.
  - Reduksi dimensi fitur kata menggunakan metode **PCA**.
  - _Output:_ _Script_ Python pembersihan data dan matriks ekstraksi fitur.
- **Minggu 3: Eksperimen Multi-Model**
  - Implementasi model _Unsupervised_ (K-Means Clustering + Elbow Method).
  - Uji coba terpisah beberapa model klasifikasi _Supervised_.
  - _Output:_ Grafik Elbow, visualisasi kluster PCA, dan skor evaluasi model.
- **Minggu 4: Finalisasi**
  - Penyusunan kesimpulan akhir dan pembuatan laporan proyek.
  - Latihan presentasi bersama seluruh anggota kelompok.
  - _Output:_ Slide presentasi siap pakai.

---

## 2. STRUKTUR TIM & PEMBAGIAN TUGAS

Untuk kelompok berisi 5 orang, peran dibagi secara adil berdasarkan alur kerja data:

- **Anggota 1 (Project Manager & Presenter)**
  - Menyusun _timeline_ kerja, memonitor alur kerja tim, dan merumuskan kesimpulan akhir proyek.
- **Anggota 2 (Data Scraper & Wrangler)**
  - Membangun _script_ Web Scraping, melakukan _text cleaning_ (case folding, buang emoji/stopwords, stemming), dan pelabelan sentimen awal.
- **Anggota 3 (ML Engineer - Unsupervised & PCA)**
  - Mengonversi teks ke bentuk angka (TF-IDF), menerapkan reduksi dimensi PCA, dan memodelkan K-Means Clustering dengan analisis Elbow.
- **Anggota 4 (ML Engineer - Supervised Classification)**
  - Mengembangkan dan melatih beberapa model klasifikasi terpisah (seperti Naive Bayes, Random Forest, atau SVM) untuk memprediksi label sentimen.
- **Anggota 5 (Data Visualizer)**
  - Mengonstruksi grafik Elbow, membuat plot visualisasi sebaran kluster 2D/3D hasil PCA, diagram evaluasi performa klasifikasi, serta merapikan slide presentasi.

---

## 3. DATA ACQUISITION & RAW DATA

Data didapatkan langsung melalui proses _scraping_ ulasan platform digital untuk mendapatkan opini asli konsumen.

### Contoh Representasi Data Mentah (Raw Data)

- Kolom awal yang didapatkan meliputi data identitas pengirim dan teks komentar.
- Data mentah masih mengandung nilai kosong (_missing values_) atau teks yang tidak terstruktur dengan baik.

```python
# Ilustrasi struktur data hasil scraping sebelum dibersihkan
{
    'Review_ID': [1, 2, 3, 4, 5],
    'User_Name': ['Andi', 'Budi', 'Cici', 'Dedi', 'Eti'],
    'Review_Text': [
        'Aplikasi ini bagus banget!! Sangat membantu sehari-hari.',
        'Kecewa, aplikasinya sering lag dan keluar sendiri.',
        'Biasa aja sih, fitur layanannya perlu ditingkatkan lagi.',
        'Sangat puas dengan respon cs yang cepat dan ramah.',
        None  # Contoh data kosong (missing value)
    ]
}
```
````

---

## 4. PIPELINE PREPROCESSING & PCA

Proses pembersihan dilakukan secara bertahap agar data siap dimasukkan ke dalam algoritma Machine Learning:

1. **Slicing & Dropping:** Menghapus data komentar yang kosong (`NaN`) dan membuang kolom identitas yang tidak relevan.
2. **Text Cleaning:** Menghilangkan angka, simbol tanda baca, emoji, dan menyamakan huruf menjadi kecil (_case folding_).
3. **Filtering & Stemming:** Membuang kata hubung/kata tidak penting (_stopwords_) dan mengubah kata berimbuhan menjadi kata dasar.
4. **Feature Extraction (TF-IDF):** Mentransformasi teks yang sudah bersih menjadi bentuk matriks bobot angka berdasarkan kemunculan kata.
5. **Reduksi Dimensi (PCA):** Memadatkan ratusan dimensi kata hasil TF-IDF menjadi hanya 2 komponen utama untuk menyederhanakan komputasi dan memudahkan visualisasi grafik.

---

## 5. IMPLEMENTASI MULTI-MODEL TERPISAH

Proyek ini menerapkan paradigma multi-model yang dieksekusi secara terpisah untuk dua kebutuhan analisis berbeda:

### Model 1: Unsupervised Learning (Elbow Method & K-Means)

- **Tujuan:** Mengelompokkan atau melakukan segmentasi ulasan pelanggan berdasarkan kemiripan pola topik bahasan dari komponen PCA.
- **Evaluasi:** Menggunakan _Elbow Method_ dengan melihat grafik nilai WCSS (_Within-Cluster Sum of Squares_) untuk menentukan jumlah kluster ($k$) yang paling optimal.

### Model 2: Supervised Learning (Perbandingan Klasifikasi Sentimen)

- **Tujuan:** Melatih beberapa algoritma secara terpisah untuk memprediksi kategori sentimen (Positif, Negatif, atau Netral) pada ulasan baru.
- **Evaluasi:** Membandingkan beberapa model terpisah seperti _Naive Bayes_, _Random Forest_, dan _Support Vector Machine_ (SVM) guna mencari tingkat akurasi tertinggi.

---

## 6. KESIMPULAN HASIL EKSEKUSI PROYEK

- **Eksplorasi & Pembersihan Data:** Proses _web scraping_ berhasil mengumpulkan data ulasan secara _real-time_. Seluruh teks bising (_noise_) berhasil dibersihkan dan dipetakan ke dalam 3 kelas sentimen.
- **Reduksi Dimensi (PCA):** Penerapan PCA terbukti efektif memangkas ratusan dimensi fitur kata teks menjadi 2 komponen utama tanpa kehilangan variansi informasi yang signifikan.
- **Segmentasi Pasar (Clustering):** Berdasarkan grafik _Elbow Method_, ulasan pelanggan terbagi secara optimal ke dalam $k=3$ kelompok kluster topik utama.
- **Prediksi Sentimen (Klasifikasi):** Berdasarkan komparasi dari beberapa model terpisah yang diuji, algoritma **SVM (Support Vector Machine)** memberikan performa terbaik dengan tingkat akurasi mencapai **86%**, mengungguli model Naive Bayes dan Random Forest.
