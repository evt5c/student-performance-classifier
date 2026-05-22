"""
train_model.py
==============
Script training model ANN/MLP untuk Student Performance Classifier.
Jalankan SEKALI sebelum menjalankan app.py.

Nama  : Steve Chenaldy Chendra
Kelas : AIB02 – Pemrograman Kecerdasan Buatan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)


# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("=" * 55)
print("  Student Performance Classifier — Training Script")
print("=" * 55)

CSV_FILE = "StudentsPerformance.csv"

if os.path.exists(CSV_FILE):
    print(f"\n[1/6] Memuat dataset dari file lokal: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
else:
    print(f"\n[1/6] File lokal tidak ditemukan. Mencoba download...")
    URLS = [
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/StudentsPerformance.csv",
        "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/StudentsPerformance.csv",
    ]
    df = None
    for url in URLS:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            print(f"    ✓ Berhasil dari: {url}")
            df.to_csv(CSV_FILE, index=False)
            print(f"    ✓ Disimpan ke {CSV_FILE}")
            break
        except Exception as e:
            print(f"    ✗ Gagal: {e}")

    if df is None:
        print("\n  ERROR: Semua sumber gagal.")
        print("  Silakan download manual dari:")
        print("  https://www.kaggle.com/datasets/spscientist/students-performance-in-exams")
        print("  Letakkan file CSV di folder yang sama dengan train_model.py")
        exit(1)

# Standarisasi nama kolom
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(' ', '_').str.replace('/', '_'))

print(f"    Dataset shape: {df.shape}")
print(f"    Kolom: {df.columns.tolist()}")


# ── 2. Buat Label Target ───────────────────────────────────────────────────────
print("\n[2/6] Membuat label performance_category...")

score_cols = ['math_score', 'reading_score', 'writing_score']
df['average_score'] = df[score_cols].mean(axis=1)

def kategorikan(avg):
    if avg >= 75: return 'High'
    elif avg >= 50: return 'Medium'
    else: return 'Low'

df['performance_category'] = df['average_score'].apply(kategorikan)

dist = df['performance_category'].value_counts()
for k, v in dist.items():
    print(f"    {k:8s}: {v} siswa")


# ── 3. Preprocessing ───────────────────────────────────────────────────────────
print("\n[3/6] Preprocessing data...")

# Fitur: gender, kursus persiapan, 3 skor ujian
feature_cols   = ['gender', 'test_preparation_course',
                  'math_score', 'reading_score', 'writing_score']
cat_cols_encode = ['gender', 'test_preparation_course']

X = df[feature_cols].copy()
y = df['performance_category'].copy()

# One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=cat_cols_encode, drop_first=False)
feature_names = X_encoded.columns.tolist()
print(f"    Fitur setelah encoding: {len(feature_names)} kolom")
print(f"    {feature_names}")

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"    Kelas: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split 70/15/15
X_arr = X_encoded.values
X_temp, X_test, y_temp, y_test = train_test_split(
    X_arr, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"    Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Normalisasi
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)


# ── 4. Bangun Model ────────────────────────────────────────────────────────────
print("\n[4/6] Membangun model ANN/MLP...")

num_classes = len(le.classes_)
input_dim   = X_train.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,),
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
model.summary()


# ── 5. Training ────────────────────────────────────────────────────────────────
print("\n[5/6] Melatih model...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15,
        restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7,
        min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_accuracy',
        save_best_only=True, verbose=0),
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print(f"\n    Training selesai di epoch: {len(history.history['loss'])}")


# ── 6. Evaluasi & Simpan ───────────────────────────────────────────────────────
print("\n[6/6] Evaluasi dan menyimpan model...")

best_model = tf.keras.models.load_model('best_model.keras')
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\n    Test Loss    : {test_loss:.4f}")
print(f"    Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5)
plt.title('Confusion Matrix', fontsize=13, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("    Confusion matrix disimpan: confusion_matrix.png")

# Learning Curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Val', linestyle='--')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history.history['accuracy'], label='Train')
axes[1].plot(history.history['val_accuracy'], label='Val', linestyle='--')
axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150)
print("    Learning curves disimpan: learning_curves.png")

# Simpan artifacts
best_model.save('student_performance_model.keras')
joblib.dump(scaler,        'scaler.pkl')
joblib.dump(le,            'label_encoder.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("\n" + "=" * 55)
print("  Semua file model berhasil disimpan:")
for f in ['student_performance_model.keras', 'scaler.pkl',
          'label_encoder.pkl', 'feature_names.pkl']:
    size = os.path.getsize(f) / 1024
    print(f"    ✓  {f}  ({size:.1f} KB)")
print("\n  Sekarang jalankan: streamlit run app.py")
print("=" * 55)
