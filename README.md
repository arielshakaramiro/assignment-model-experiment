# AI Model Experiment & Evaluation: Sentiment Analysis Ulasan Pelanggan

**Model Klasik (Scikit-learn) vs LLM API (Gemini)**
Individual Assignment, AI Engineering Bootcamp · Muhammad Ariel Shakaramiro

![Ringkasan perbandingan](documentation/model_comparison_summary.png)

> **Ringkasan:** Pada test set resmi, model klasik dan Gemini sama-sama mendapat skor sempurna (F1 1.00). Namun 98% kalimat di test set ternyata sudah ada di train set. Saat diuji pada kalimat yang benar-benar baru, model klasik turun ke F1 0.52–0.63, sedangkan Gemini tetap 1.00, dan Gemini juga benar 8/8 pada ulasan sulit (typo, sarkasme, campuran, bahasa Inggris). **Rekomendasi: kembangkan Gemini zero-shot dengan arsitektur asinkron + caching**, lalu siapkan jalur distilasi ke model sendiri.

---

## 1. Problem Statement

**Objective.** Tim produk e-commerce ingin fitur otomatis untuk mengklasifikasikan sentimen ulasan pelanggan pada halaman produk. Eksperimen ini membandingkan dua kandidat pendekatan, yaitu model klasik yang dilatih sendiri dan LLM API berbasis prompting, lalu menghasilkan **rekomendasi berbasis data** tentang pendekatan yang layak dikembangkan ke produksi. Pertimbangannya mencakup performa, effort implementasi, kecepatan, dan biaya.

**Target/label.** Klasifikasi biner kolom `sentiment`:

| Label | Definisi |
|---|---|
| `positif` | Pelanggan puas atau memuji produk, pengiriman, atau pelayanan |
| `negatif` | Pelanggan kecewa, mengeluh, atau melaporkan masalah |

Input model hanya `review_text`. `product_name` tidak dipakai karena sentimen ditentukan oleh isi ulasan.

**Batasan & asumsi.**
- Memakai dataset resmi assignment tanpa modifikasi. Label dianggap *ground truth*.
- Hanya dua kelas. Ulasan netral atau campuran berada di luar cakupan.
- Tidak ada training deep learning dari nol.
- Biaya API diestimasi dari token aktual × harga publik Gemini API (Oktober 2026). Harga bisa berubah.

## 2. Dataset

`data/customer_reviews_sentiment.csv`: 200 ulasan berbahasa Indonesia, kolom `review_id`, `product_name`, `review_text`, `sentiment`.

| Karakteristik | Nilai |
|---|---|
| Jumlah baris | 200 (positif 110 / negatif 90, cukup seimbang) |
| Missing value | 0 |
| **Teks ulasan unik** | **hanya 40** (20 positif, 20 negatif), masing-masing berulang 1–12× |
| Label bertentangan untuk teks sama | 0 |
| Panjang ulasan | 6–11 kata |

**Temuan kunci: data leakage.** Karena hanya ada 40 kalimat unik, split acak per baris (80/20, `random_state=42`, stratified) menghasilkan test set yang **39 dari 40 barisnya (98%) identik dengan kalimat di train set**. Karena itu eksperimen ini memakai dua skema evaluasi:

1. **Split utama** (sesuai brief/starter): perbandingan resmi, test set **sama** untuk kedua pendekatan (`results/test_set.csv`).
2. **Split leakage-free** (`GroupShuffleSplit` per teks unik + 5-fold `GroupKFold` CV): mengukur generalisasi pada kalimat yang belum pernah dilihat. Test set ini juga dipakai untuk **kedua** pendekatan.

## 3. Ringkasan Eksperimen

### Pendekatan 1: Model Klasik (Scikit-learn)
- **Fitur:** `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)`. Bigram menangkap negasi seperti *"tidak sesuai"*. Stopword tidak dihapus karena *"tidak"* membawa sinyal sentimen. Hasilnya 384 fitur.
- **Model:** `LogisticRegression(max_iter=1000)`, cepat dan interpretable. Bobot terkuat: *lambat, jauh dari, rusak, tidak sesuai* (negatif) vs *cepat, rapi, puas, recommended* (positif).
- **Baseline:** `DummyClassifier` kelas mayoritas.

### Pendekatan 2: LLM API (Gemini)
- **Model:** `gemini-3.1-flash-lite`. `gemini-2.0-flash` pada starter sudah dihentikan Google (1 Juni 2026), dan kuota harian free tier `gemini-3.8-flash` habis di tengah eksperimen. Agar konsisten, **semua prediksi LLM berasal dari satu model**.
- **Prompt:** system instruction berisi peran, definisi tiap label, peringatan negasi, dan aturan "jawab satu kata". Diuji dua strategi:
  - **Zero-shot:** tanpa contoh.
  - **Few-shot:** + 4 contoh berlabel (2 positif, 2 negatif) dari train set yang **dipastikan tidak ada di test set mana pun**.
- **Parameter:**

| Parameter | Nilai | Alasan |
|---|---|---|
| `temperature` | 0.0 | Klasifikasi butuh jawaban deterministik & reproducible, bukan kreativitas |
| `thinking_level` | LOW | Tugas sederhana, sehingga latensi dan token "berpikir" bisa ditekan |
| `response_mime_type` + enum schema | `positif`/`negatif` | Membatasi output agar selalu valid. Hasilnya 0 output invalid. |
| `seed` | 42 | Reproducibility |

- **Pemanggilan API:** satu panggilan per teks unik, hasilnya disimpan di `results/llm_cache.json`. Ada jeda antar panggilan karena rate limit free tier, dan error 429/503 dicoba ulang dengan jeda yang makin lama.

## 4. Hasil Evaluasi & Perbandingan

### 4.1 Test set utama (n = 40, sama untuk semua pendekatan)

| Pendekatan | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Recall negatif | Latensi / ulasan |
|---|---|---|---|---|---|---|
| Baseline (kelas mayoritas) | 0.550 | 0.275 | 0.500 | 0.355 | 0.000 | - |
| TF-IDF + Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ±0,02 ms |
| Gemini zero-shot | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6,5 s (median) |
| Gemini few-shot | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7,6 s (median) |

**Confusion matrix:** identik untuk ketiga pendekatan.

| | pred negatif | pred positif |
|---|---|---|
| aktual negatif | 18 | 0 |
| aktual positif | 0 | 22 |

**Interpretasi.**
- **Accuracy 1.00:** semua 40 ulasan benar, jauh di atas baseline 0.55.
- **Precision 1.00:** tidak ada *false alarm*, tidak ada ulasan positif yang ditandai negatif.
- **Recall negatif 1.00:** semua keluhan tertangkap. Ini metrik bisnis terpenting.
- **F1 1.00:** precision dan recall seimbang sempurna.

Namun skor seri ini **tidak bisa dipakai untuk memilih pendekatan**. Bagi model klasik, 1.00 adalah artefak leakage (ia sudah "melihat" 98% kalimat test). Bagi Gemini, skor ini murni, karena Gemini tidak pernah dilatih pada data ini. Few-shot juga tidak menambah akurasi dibanding zero-shot, padahal token input naik ±76%.

### 4.2 Evaluasi leakage-free (kalimat test tidak pernah dilihat saat training)

| Pendekatan | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Recall negatif |
|---|---|---|---|---|---|
| TF-IDF + LogReg (GroupShuffleSplit, n=41 / 8 teks unik) | 0.634 | 0.741 | 0.722 | 0.633 | 1.000 |
| TF-IDF + LogReg (5-fold GroupKFold, rata-rata) | 0.540 ± 0.15 | 0.656 | 0.658 | 0.524 ± 0.16 | - |
| **Gemini zero-shot** (GroupShuffleSplit) | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| **Gemini few-shot** (GroupShuffleSplit) | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

**Interpretasi.** Model klasik **runtuh** ke tingkat hampir menebak (accuracy 0.54 di CV, baseline 0.55). Recall negatifnya memang 1.00, tetapi precision negatifnya hanya 0.48: **15 dari 27 ulasan positif salah ditandai negatif** (lihat confusion matrix pada grafik di atas). Model "menuduh" negatif setiap bertemu kata yang tidak dikenal. Gemini tetap sempurna (41/41). Catatan: test set ini hanya berisi 8 kalimat unik, jadi angka 1.00 punya ketidakpastian lebar, tapi arahnya konsisten dengan uji ketahanan.

### 4.3 Uji ketahanan (8 ulasan buatan di luar dataset, ilustratif)

| Tantangan | Contoh | Label | Klasik | Gemini few-shot |
|---|---|---|---|---|
| negasi | Tidak mengecewakan sama sekali, bakal repeat order. | positif | benar (p=0.54) | benar |
| slang + negasi | Gak nyesel beli di sini, recommended bgt | positif | benar | benar |
| sarkasme | Wah hebat, baru sehari dipakai udah rusak. Mantap. | negatif | benar (p=0.45) | benar |
| campuran | Barangnya bagus sih, tapi pengirimannya lama banget dan kurirnya kasar. | negatif | salah (positif) | benar |
| ekspektasi terbalik | Kirain bakal jelek, ternyata kualitasnya bagus banget. | positif | benar | benar |
| singkatan/typo | brg gk sesuai pesanan, admin jg slow resp | negatif | salah (positif) | benar |
| kontras | Murah tapi bukan murahan, puas! | positif | salah (negatif) | benar |
| bahasa Inggris | Not recommended, quality is very poor. | negatif | salah (positif) | benar |
| **Total** | | | **4/8** | **8/8** |

Model klasik hanya mengenali **kata** yang pernah dilihat, bukan **makna** kalimat. Bahkan prediksinya yang benar hanya punya probabilitas 0.45–0.65.

### 4.4 Kecepatan & biaya

| | Model klasik | Gemini zero-shot | Gemini few-shot |
|---|---|---|---|
| Effort awal | Butuh data berlabel + training | Tanpa training, cukup prompt | Prompt + 4 contoh |
| Latensi/ulasan | ±0,02 ms (CPU lokal) | median 6,5 s · p95 22 s | median 7,6 s · p95 93 s |
| Token/ulasan (input / output+thinking) | - | 140 / 112 | 246 / 110 |
| Biaya eksperimen (26 panggilan) | Rp0 | US$0,0053 | US$0,0059 |
| Proyeksi 1 juta ulasan | ≈ Rp0 | ±US$203 (≈ Rp3,3 juta) | ±US$226 (≈ Rp3,7 juta) |

Harga `gemini-3.1-flash-lite`: US$0,25 / 1 juta token input dan US$1,50 / 1 juta token output (termasuk thinking). Sekitar **83% biaya output adalah thinking token** (±110 dari 112 token), padahal jawabannya hanya 2 token. Latensi Gemini terukur saat server Google sedang padat (banyak error 503), sehingga lebih tinggi dari kondisi normal.

## 5. Analisis Trade-off & Limitation

**Performa.** Di test set resmi keduanya seri sempurna. Di kondisi yang mencerminkan produksi (kalimat baru, bahasa tidak baku), **Gemini jelas unggul**: F1 1.00 vs 0.52–0.63, dan 8/8 vs 4/8 pada ulasan sulit.

**Effort implementasi.** Model klasik butuh dataset berlabel besar dan beragam, dan 40 kalimat unik terbukti tidak cukup. Gemini tidak butuh training sama sekali, tapi butuh effort operasional: API key, retry, caching, pemantauan kuota, dan antisipasi pergantian model.

**Kecepatan & biaya.** Model klasik ratusan ribu kali lebih cepat dan praktis gratis. Gemini butuh detik per ulasan dan biayanya naik linear dengan volume (±Rp3,3 juta per 1 juta ulasan).

**Limitation model klasik:**
- Menghafal, bukan memahami. Skor 1.00 hanya karena leakage.
- Vocabulary tertutup (384 fitur), sehingga gagal pada typo, singkatan, dan bahasa Inggris.
- Bias kata permukaan: "bagus" membuat keluhan campuran dianggap positif, "murahan" membuat pujian dianggap negatif.
- Variansi tinggi antar-fold (±0.15) karena data latih terlalu sedikit.

**Limitation Gemini (banyak yang dialami langsung saat eksperimen):**
- **Ketergantungan pada layanan eksternal:** kuota harian free tier `gemini-3.8-flash` habis setelah ±15 panggilan (429), project sempat diblokir (403), `gemini-2.5-flash` tidak tersedia untuk project baru (404), server padat (503), dan model di starter (`gemini-2.0-flash`) sudah dihentikan.
- **Latensi tinggi dan tidak stabil** (median ±7 s, terburuk >90 s), tidak cocok untuk klasifikasi sinkron.
- **Biaya per panggilan**, didominasi thinking token.
- **Privasi:** teks ulasan dikirim ke pihak ketiga.
- **Evaluasi masih terbatas:** dataset berisi kalimat eksplisit dan pendek. Kesalahan prediksi Gemini belum teramati, sehingga pola kegagalannya pada ulasan nyata yang panjang/ambigu/netral belum diketahui.

## 6. Rekomendasi Technical Approach

### Kembangkan **LLM API (Gemini, zero-shot)** sebagai pendekatan utama, dengan arsitektur asinkron + caching, dan siapkan jalur distilasi ke model sendiri.

**Alasan:**
1. **Satu-satunya pendekatan yang terbukti generalisasi.** Di produksi hampir semua ulasan adalah kalimat baru, jadi yang relevan adalah skor leakage-free (Gemini 1.00 vs klasik 0.52–0.63), bukan skor 1.00 di split resmi.
2. **Tahan terhadap bahasa ulasan nyata** (typo, slang, sarkasme, campuran, bahasa Inggris): 8/8 vs 4/8.
3. **Tidak butuh dataset berlabel besar**, yang memang belum tersedia.
4. **Zero-shot cukup.** Few-shot tidak menambah akurasi tapi menambah ±76% token input.

**Mitigasi kelemahan LLM:**
- **Asinkron:** klasifikasikan saat ulasan dikirim (background job/queue) dan simpan labelnya. Halaman produk hanya membaca label, sehingga latensi tidak terasa pengguna.
- **Caching per teks:** 80% baris di dataset ini adalah duplikat. Dengan `temperature=0`, ulasan identik cukup diklasifikasikan sekali.
- **Paid tier + retry/backoff + fallback model**, dan pantau jadwal deprecation model.
- **`thinking_level=MINIMAL`** untuk memangkas ±83% biaya output (validasi ulang akurasinya).
- **Gagal = tunda, bukan tebak:** ulasan yang gagal diklasifikasi ditandai "pending" dan diproses ulang.

**Jangka menengah: hybrid/distilasi.** Gunakan label Gemini (plus validasi manual pada sampel) untuk membangun dataset ribuan ulasan nyata. Latih model sendiri (TF-IDF+LogReg atau fine-tune IndoBERT) dan evaluasi dengan split leakage-free. Jika setara, model sendiri mengambil alih traffic volume tinggi (biaya ≈ Rp0, latensi milidetik), dan LLM dipakai untuk kasus ber-confidence rendah.

**Kapan rekomendasi berubah:** model klasik lebih tepat jika tersedia dataset berlabel besar, volume sangat tinggi, dibutuhkan respons real-time di bawah 100 ms, atau kebijakan privasi melarang pengiriman data ke pihak ketiga.

## 7. Cara Menjalankan

```bash
# 1. Clone & masuk folder
git clone <url-repo> && cd model-experiment-assignment

# 2. Buat virtual environment & install dependency
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Opsional) Siapkan API key Gemini untuk memanggil API ulang
cp .env.example .env               # isi GEMINI_API_KEY, dan GEMINI_MODEL=gemini-3.1-flash-lite

# 4. Jalankan notebook
jupyter notebook notebook/experiment_notebook.ipynb   # atau buka di VS Code -> Run All
```

**Catatan:**
- **Tanpa API key**, notebook tetap berjalan tanpa error dan mereproduksi seluruh hasil memakai respons Gemini yang tersimpan di `results/llm_cache.json`.
- Untuk memanggil API ulang (misal setelah mengubah prompt), hapus `results/llm_cache.json`.
- Atur kecepatan panggilan dengan `GEMINI_RPM` (default 5 request/menit) sesuai kuota akun.
- API key **tidak pernah** di-hardcode, dan `.env` sudah masuk `.gitignore`.

## 8. Struktur Project

```
model-experiment-assignment/
├── data/
│   └── customer_reviews_sentiment.csv     # dataset asli (tidak diubah)
├── notebook/
│   └── experiment_notebook.ipynb          # seluruh eksperimen, evaluasi, analisis & rekomendasi
├── documentation/
│   └── model_comparison_summary.png       # grafik ringkasan perbandingan
├── results/                               # output eksperimen (dibuat otomatis oleh notebook)
│   ├── test_set.csv                       # test set utama yang dipakai kedua pendekatan
│   ├── predictions_test_set.csv           # prediksi per baris kedua pendekatan
│   ├── model_comparison.csv               # tabel ringkasan (split utama)
│   ├── model_comparison_leakage_free.csv  # tabel ringkasan (leakage-free)
│   ├── robustness_probe.csv               # uji ketahanan
│   ├── llm_cost_latency.csv               # token, latensi, biaya Gemini
│   ├── llm_cache.json                     # respons mentah Gemini
│   └── metrics_summary.json
├── .env.example
├── README.md
└── requirements.txt
```
