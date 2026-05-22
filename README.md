# 🎓 Student Performance Category Classifier

Aplikasi klasifikasi performa akademik siswa menggunakan **Artificial Neural Network (ANN/MLP)** berbasis Keras.  
Model memprediksi kategori performa siswa — **Tinggi**, **Sedang**, atau **Rendah** — berdasarkan nilai ujian dan data diri.

> **Nama:** Steve Chenaldy Chendra  
> **Mata Kuliah:** AIB02 – Pemrograman Kecerdasan Buatan  
> **Universitas Bunda Mulia

---

## 📁 Struktur Folder

```
student-performance-classifier/
├── train_model.py                    ← Script training model (jalankan pertama kali)
├── app.py                            ← Aplikasi Streamlit
├── requirements.txt                  ← Daftar dependensi Python
├── README.md                         ← File ini
├── StudentsPerformance.csv           ← Dataset (auto-download jika tidak ada)
│
│   (dihasilkan setelah train_model.py dijalankan)
├── student_performance_model.keras
├── scaler.pkl
├── label_encoder.pkl
└── feature_names.pkl
```

---

## ⚙️ Persyaratan

- **Python 3.9 – 3.11** (disarankan Python 3.11)
- pip

> ⚠️ TensorFlow 2.15 belum mendukung Python 3.12 ke atas.  
> Download Python 3.11 di [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/)

---

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/USERNAME/student-performance-classifier.git
cd student-performance-classifier
```

### 2. Buat virtual environment

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Mac/Linux
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan training model

```bash
python train_model.py
```

Script ini akan otomatis:
- Mendownload dataset jika belum ada
- Melatih model ANN/MLP
- Menyimpan file model (`.keras` dan `.pkl`)
- Menampilkan hasil evaluasi (accuracy, classification report)

Tunggu hingga muncul pesan:
```
Sekarang jalankan: streamlit run app.py
```

### 5. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

Buka browser di **http://localhost:8501**

---

## 🧠 Tentang Model

| Komponen | Detail |
|---|---|
| **Arsitektur** | MLP 3 hidden layer |
| **Hidden Layer** | Dense(128) → Dense(64) → Dense(32) |
| **Aktivasi** | ReLU (hidden), Softmax (output) |
| **Regularisasi** | Dropout + Batch Normalization + L2 |
| **Loss** | Sparse Categorical Crossentropy |
| **Optimizer** | Adam (lr=0.001) |
| **Output** | 3 kelas: Low / Medium / High |

---

## 📊 Dataset

| Atribut | Keterangan |
|---|---|
| **Nama** | Students Performance in Exams |
| **Sumber** | [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) |
| **Jumlah Data** | 1.000 baris |
| **Fitur yang Digunakan** | Jenis Kelamin, Kursus Persiapan Ujian, Nilai Matematika, Nilai Membaca, Nilai Menulis |
| **Label** | `performance_category` (Low / Medium / High) |

---

## 🖥️ Tampilan Aplikasi

Input yang tersedia:
- **Jenis Kelamin** — Perempuan / Laki-laki
- **Kursus Persiapan Ujian** — Sudah Mengikuti / Tidak Mengikuti
- **Nilai Matematika** — skala 0–100
- **Nilai Membaca** — skala 0–100
- **Nilai Menulis** — skala 0–100

Output:
- Kategori performa: 🟢 Tinggi / 🟡 Sedang / 🔴 Rendah
- Tingkat keyakinan model (confidence %)
- Distribusi probabilitas tiap kategori

---

## 🔧 Troubleshooting

**`py -3.11` tidak ditemukan (Windows)**  
→ Download dan install Python 3.11 dari [python.org](https://www.python.org/downloads/release/python-3119/), centang "Add to PATH" saat install.

**Dataset gagal didownload otomatis**  
→ Download manual dari [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams), letakkan file `StudentsPerformance.csv` di folder project.

**`streamlit` tidak dikenali**  
→ Pastikan virtual environment sudah aktif (ada tulisan `(venv)` di terminal).

**File `.keras` atau `.pkl` tidak ditemukan saat buka app**  
→ Jalankan `python train_model.py` terlebih dahulu.
