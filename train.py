"""
Full training pipeline for the IDS system.
Run from the ids-sytem directory: python train.py
"""
import os
import sys
import zipfile
import time

# Force UTF-8 output so Unicode chars (including � in attack labels) don't crash on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
ARCHIVE_ZIP = Path(r"C:/Users/Tero G/Desktop/Tero Final Year Project/archive.zip")
EXTRACT_DIR = ROOT / "cicids2017"
MODELS_DIR  = ROOT / "models"
SCALERS_DIR = ROOT / "scalers"
DATA_DIR    = ROOT / "data" / "checkpoints"

for d in [MODELS_DIR, SCALERS_DIR, DATA_DIR, EXTRACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Step 1: Extract archive ────────────────────────────────────────────────────
def extract_data():
    parquet_files = list(EXTRACT_DIR.glob("*.parquet"))
    if parquet_files:
        print(f"[1/5] Using already-extracted parquet files ({len(parquet_files)} files)")
        return parquet_files

    print(f"[1/5] Extracting {ARCHIVE_ZIP} ...")
    with zipfile.ZipFile(ARCHIVE_ZIP, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    parquet_files = list(EXTRACT_DIR.glob("*.parquet"))
    print(f"      Extracted {len(parquet_files)} parquet files.")
    return parquet_files

# ── Step 2: Load & clean data ──────────────────────────────────────────────────
def load_and_process(parquet_files):
    ckpt = DATA_DIR / "processed_cicids_data.parquet"
    if ckpt.exists():
        print(f"[2/5] Loading from checkpoint (skipping re-processing) ...")
        t0 = time.time()
        df = pd.read_parquet(ckpt)
        print(f"      Loaded {df.shape[0]:,} rows x {df.shape[1]} cols from checkpoint in {time.time()-t0:.1f}s")
        return df

    print("[2/5] Loading and processing data ...")
    t0 = time.time()

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"      Loaded {df.shape[0]:,} rows × {df.shape[1]} cols in {time.time()-t0:.1f}s")

    # Clean column names
    df.columns = (df.columns.str.strip()
                              .str.replace(' ', '_')
                              .str.replace('/', '_')
                              .str.replace('.', '', regex=False))

    # Preserve original string label
    df['Attack_Type_Original'] = df['Label'].astype(str).str.strip()

    # Binary label
    df['Label'] = df['Label'].astype(str).str.strip().str.lower()
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'benign' else 1)

    # Numeric coercion for feature columns
    for col in df.columns:
        if col not in ('Label', 'Attack_Type_Original') and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)

    # Impute
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if not mode.empty else 'unknown')

    print(f"      Final shape: {df.shape}  |  Label dist: {dict(df['Label'].value_counts())}")

    # Checkpoint
    ckpt = DATA_DIR / "processed_cicids_data.parquet"
    df.to_parquet(ckpt, index=False)
    print(f"      Saved checkpoint -> {ckpt}")
    return df

# ── Step 3: Binary classifiers ────────────────────────────────────────────────
def train_binary(df):
    print("[3/5] Training binary classifiers (RF + XGBoost) ...")
    X = df.drop(['Label', 'Attack_Type_Original'], axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    # Random Forest
    print("      Training Random Forest (n_estimators=50) ...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"      RF trained in {time.time()-t0:.1f}s")
    _eval_binary(rf, X_test, y_test, "Random Forest")
    rf_path = MODELS_DIR / "random_forest_binary_classifier.joblib"
    joblib.dump(rf, rf_path)
    print(f"      Saved -> {rf_path}")

    # XGBoost
    print("      Training XGBoost (n_estimators=100) ...")
    t0 = time.time()
    xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                         n_estimators=100, learning_rate=0.1,
                         random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    print(f"      XGB trained in {time.time()-t0:.1f}s")
    _eval_binary(xgb, X_test, y_test, "XGBoost")
    xgb_path = MODELS_DIR / "xgboost_binary_classifier.joblib"
    joblib.dump(xgb, xgb_path)
    print(f"      Saved -> {xgb_path}")

    return rf, xgb, X_test, y_test

def _eval_binary(model, X_test, y_test, name):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n      -- {name} --")
    print(f"        Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"        Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"        Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"        F1       : {f1_score(y_test, y_pred):.4f}")
    print(f"        ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

# ── Step 4: LSTM multi-class ──────────────────────────────────────────────────
def train_lstm(df):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    print("[4/5] Training LSTM multi-class classifier ...")

    attack_df = df[df['Label'] == 1].copy().drop('Label', axis=1)
    X_att = attack_df.drop('Attack_Type_Original', axis=1).fillna(0)
    y_att = attack_df['Attack_Type_Original']

    print(f"      Attack samples: {len(attack_df):,}  |  Classes: {y_att.nunique()}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_att)
    joblib.dump(le, SCALERS_DIR / "attack_type_label_encoder.joblib")
    print(f"      LabelEncoder saved  ({len(le.classes_)} classes)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_att)
    joblib.dump(scaler, SCALERS_DIR / "attack_type_scaler.joblib")
    print(f"      Scaler saved")

    y_oh = to_categorical(y_enc)
    num_classes = y_oh.shape[1]
    num_features = X_scaled.shape[1]

    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, num_features)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_reshaped, y_oh, test_size=0.3, random_state=42, stratify=y_enc)

    model = Sequential([
        LSTM(100, activation='relu', input_shape=(1, num_features)),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    t0 = time.time()
    history = model.fit(X_tr, y_tr, epochs=20, batch_size=64,
                        validation_split=0.2, callbacks=[es], verbose=1)
    print(f"      LSTM trained in {time.time()-t0:.1f}s")

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"      Test Loss: {loss:.4f}  |  Test Acc: {acc:.4f}")

    y_pred_enc = np.argmax(model.predict(X_te, verbose=0), axis=1)
    y_true_enc = np.argmax(y_te, axis=1)
    print("\n" + classification_report(y_true_enc, y_pred_enc, target_names=le.classes_,
                                       labels=range(len(le.classes_))))

    lstm_path = MODELS_DIR / "lstm_multi_class_classifier.h5"
    model.save(lstm_path)
    print(f"      Saved -> {lstm_path}")
    return model, le

# ── Step 5: Verify all artifacts ─────────────────────────────────────────────
def verify():
    print("[5/5] Verifying saved artifacts ...")
    required = [
        MODELS_DIR  / "random_forest_binary_classifier.joblib",
        MODELS_DIR  / "xgboost_binary_classifier.joblib",
        MODELS_DIR  / "lstm_multi_class_classifier.h5",
        SCALERS_DIR / "attack_type_scaler.joblib",
        SCALERS_DIR / "attack_type_label_encoder.joblib",
    ]
    all_ok = True
    for p in required:
        if p.exists():
            print(f"      OK  {p.name}  ({p.stat().st_size:,} bytes)")
        else:
            print(f"      MISSING  {p}")
            all_ok = False
    if all_ok:
        print("\n  All artifacts present. Run: streamlit run models/streamlit_app.py")
    return all_ok

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  IDS Training Pipeline")
    print("=" * 60)
    t_start = time.time()

    parquet_files = extract_data()
    df = load_and_process(parquet_files)
    train_binary(df)
    train_lstm(df)
    verify()

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    print("Done.")
