"""
Student Performance Category Classifier
Nama  : Steve Chenaldy Chendra
Kelas : AIB02 – Pemrograman Kecerdasan Buatan
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

# ── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Performa Siswa",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.judul { font-size: 2.1rem; font-weight: 700; color: #1e293b; line-height: 1.2; margin-bottom: 2px; }
.subjudul { color: #64748b; font-size: 0.9rem; margin-bottom: 20px; }

.kartu-hasil { border-radius: 16px; padding: 28px 30px; margin-top: 20px; text-align: center; }
.hasil-tinggi  { background: linear-gradient(135deg, #d1fae5, #a7f3d0); border-left: 6px solid #10b981; }
.hasil-sedang  { background: linear-gradient(135deg, #fef3c7, #fde68a); border-left: 6px solid #f59e0b; }
.hasil-rendah  { background: linear-gradient(135deg, #fee2e2, #fecaca); border-left: 6px solid #ef4444; }

.label-hasil   { font-size: 2.5rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
.sub-hasil     { font-size: 0.95rem; margin-top: 8px; color: #374151; }
.conf-text     { font-size: 0.88rem; margin-top: 6px; color: #6b7280; }

.seksi { font-size: 1rem; font-weight: 600; color: #1e293b; margin-bottom: 4px;
         padding-bottom: 6px; border-bottom: 2px solid #e2e8f0; }

.kotak-info { background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 0 10px 10px 0;
              padding: 10px 14px; font-size: 0.88rem; color: #1e40af; line-height: 1.6; }

.pill { display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin: 2px; }
.pill-tinggi { background: #d1fae5; color: #065f46; }
.pill-sedang { background: #fef3c7; color: #92400e; }
.pill-rendah { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def muat_model():
    required = ['student_performance_model.keras', 'scaler.pkl',
                'label_encoder.pkl', 'feature_names.pkl']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    model      = tf.keras.models.load_model('student_performance_model.keras')
    scaler     = joblib.load('scaler.pkl')
    le         = joblib.load('label_encoder.pkl')
    feat_names = joblib.load('feature_names.pkl')
    return model, scaler, le, feat_names, []

model, scaler, le, feature_names, missing_files = muat_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="judul">🎓 Klasifikasi Performa Siswa</p>', unsafe_allow_html=True)
st.markdown('<p class="subjudul">AIB02 · ANN/MLP · Steve Chenaldy Chendra</p>', unsafe_allow_html=True)

if missing_files:
    st.error(
        f"⚠️ File model tidak ditemukan: **{', '.join(missing_files)}**\n\n"
        "Jalankan notebook terlebih dahulu untuk melatih model."
    )
    st.stop()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📌 Tentang Aplikasi")
    st.markdown(
        "Aplikasi ini memprediksi **kategori performa akademik siswa** "
        "berdasarkan data diri dan nilai ujian, menggunakan model "
        "**Artificial Neural Network (ANN)** yang dibangun dengan Keras."
    )
    st.divider()
    st.markdown("### 🏷️ Kategori Performa")
    st.markdown("""
    <span class='pill pill-tinggi'>🟢 Tinggi</span> Rata-rata nilai ≥ 75<br><br>
    <span class='pill pill-sedang'>🟡 Sedang</span> Rata-rata nilai 50 – 74<br><br>
    <span class='pill pill-rendah'>🔴 Rendah</span> Rata-rata nilai &lt; 50
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📊 Dataset")
    st.markdown(
        "Students Performance in Exams  \n"
        "[Kaggle Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)"
    )
    st.divider()
    st.caption("© Steve Chenaldy Chendra · Universitas Bunda Mulia")


# ── Form Input ─────────────────────────────────────────────────────────────────
st.markdown('<p class="seksi">📋 Data Diri Siswa</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Jenis Kelamin",
        options=["female", "male"],
        format_func=lambda x: "👧 Perempuan" if x == "female" else "👦 Laki-laki",
        help="Pilih jenis kelamin siswa"
    )

with col2:
    prep = st.selectbox(
        "Kursus Persiapan Ujian",
        options=["none", "completed"],
        format_func=lambda x: "❌ Tidak Mengikuti" if x == "none" else "✅ Sudah Mengikuti",
        help="Apakah siswa mengikuti kursus persiapan ujian sebelumnya?"
    )

st.markdown("")
st.markdown('<p class="seksi">📝 Nilai Ujian Siswa</p>', unsafe_allow_html=True)
st.caption("Masukkan nilai ujian siswa (skala 0 – 100)")

col3, col4, col5 = st.columns(3)
with col3:
    math_score    = st.number_input("🔢 Matematika",    min_value=0, max_value=100, value=65,
                                    help="Nilai ujian matematika (0–100)")
with col4:
    reading_score = st.number_input("📖 Membaca",       min_value=0, max_value=100, value=68,
                                    help="Nilai ujian membaca (0–100)")
with col5:
    writing_score = st.number_input("✏️ Menulis",       min_value=0, max_value=100, value=66,
                                    help="Nilai ujian menulis (0–100)")

# Preview rata-rata
avg = (math_score + reading_score + writing_score) / 3
kat_prev = "Tinggi 🟢" if avg >= 75 else ("Sedang 🟡" if avg >= 50 else "Rendah 🔴")
st.markdown(f"""
<div class="kotak-info">
    💡 <b>Preview nilai rata-rata:</b> {avg:.1f} &nbsp;→&nbsp;
    Estimasi kategori: <b>{kat_prev}</b>
    &nbsp;(Rendah &lt;50 · Sedang 50–74 · Tinggi ≥75)
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ── Tombol Prediksi ────────────────────────────────────────────────────────────
tombol = st.button("🔍 Prediksi Performa Siswa", use_container_width=True, type="primary")

if tombol:
    # Bangun dataframe input
    input_df = pd.DataFrame([{
        'gender':                   gender,
        'test_preparation_course':  prep,
        'math_score':               math_score,
        'reading_score':            reading_score,
        'writing_score':            writing_score,
    }])

    # One-Hot Encoding sesuai training
    cat_cols = ['gender', 'test_preparation_course']
    input_enc = pd.get_dummies(input_df, columns=cat_cols, drop_first=False)

    # Selaraskan kolom dengan fitur training
    for col in feature_names:
        if col not in input_enc.columns:
            input_enc[col] = 0
    input_enc = input_enc[feature_names]

    # Normalisasi & prediksi
    X_scaled = scaler.transform(input_enc.values)
    proba    = model.predict(X_scaled, verbose=0)[0]
    pred_idx = np.argmax(proba)
    pred_eng = le.inverse_transform([pred_idx])[0]   # "High"/"Medium"/"Low"
    conf     = proba[pred_idx] * 100

    # Terjemahan hasil ke Bahasa Indonesia
    terjemahan = {'High': 'Tinggi', 'Medium': 'Sedang', 'Low': 'Rendah'}
    emoji_map  = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴'}
    css_map    = {'High': 'hasil-tinggi', 'Medium': 'hasil-sedang', 'Low': 'hasil-rendah'}
    deskripsi  = {
        'High':   'Siswa ini menunjukkan performa akademik yang sangat baik! 🎉',
        'Medium': 'Performa siswa ini cukup baik dan masih bisa ditingkatkan. 💪',
        'Low':    'Siswa ini memerlukan perhatian dan bimbingan belajar lebih lanjut. 📚'
    }

    label_id = terjemahan[pred_eng]
    emoji    = emoji_map[pred_eng]
    css_cls  = css_map[pred_eng]
    desk     = deskripsi[pred_eng]

    # ── Tampilkan Hasil ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kartu-hasil {css_cls}">
        <p class="label-hasil">{emoji} Performa: {label_id}</p>
        <p class="sub-hasil">{desk}</p>
        <p class="conf-text">Tingkat keyakinan model: <b>{conf:.1f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Distribusi probabilitas semua kelas
    st.markdown("#### 📊 Distribusi Probabilitas Tiap Kategori")
    cols = st.columns(3)
    urutan = ['Low', 'Medium', 'High']
    label_id_map = {'Low': 'Rendah 🔴', 'Medium': 'Sedang 🟡', 'High': 'Tinggi 🟢'}

    # Buat dict proba berdasarkan kelas
    proba_dict = {cls: prob for cls, prob in zip(le.classes_, proba)}

    for col_w, kls in zip(cols, urutan):
        p = proba_dict.get(kls, 0.0)
        with col_w:
            st.metric(label=label_id_map[kls], value=f"{p*100:.1f}%")
            st.progress(float(p))

    # Ringkasan input
    with st.expander("📄 Lihat Ringkasan Data Input"):
        gender_id = "Perempuan" if gender == "female" else "Laki-laki"
        prep_id   = "Sudah Mengikuti" if prep == "completed" else "Tidak Mengikuti"
        ringkasan = {
            "Jenis Kelamin":          gender_id,
            "Kursus Persiapan Ujian": prep_id,
            "Nilai Matematika":       math_score,
            "Nilai Membaca":          reading_score,
            "Nilai Menulis":          writing_score,
            "Rata-rata Nilai":        f"{avg:.1f}",
            "Hasil Prediksi":         f"{emoji} {label_id}",
            "Tingkat Keyakinan":      f"{conf:.1f}%",
        }
        st.dataframe(
            pd.DataFrame(list(ringkasan.items()), columns=["Keterangan", "Nilai"]),
            use_container_width=True, hide_index=True
        )
