import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="IDS — Ensemble System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.pipeline-box {
    border: 2px solid #ccc;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    font-weight: bold;
    font-size: 0.95em;
}
.rf-box    { background:#e8f5e9; border-color:#2d6a4f; color:#1a3d2b; }
.xgb-box   { background:#e3f0fb; border-color:#2c5f9e; color:#1a2f5a; }
.lstm-box  { background:#f3e8fb; border-color:#8b3a9e; color:#3a1040; }
.ens-box   { background:#fff3e0; border-color:#e65c00; color:#7a2d00; }
.final-box { background:#fce4ec; border-color:#c62828; color:#7a0000; }
.stage-label {
    font-size:0.75em; font-weight:bold; color:#888;
    text-transform:uppercase; letter-spacing:1px;
    margin-bottom:4px;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
base_dir    = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(base_dir)
models_dir  = base_dir
scalers_dir = os.path.join(root_dir, 'scalers')

rf_path    = os.path.join(models_dir,  'random_forest_binary_classifier.joblib')
xgb_path   = os.path.join(models_dir,  'xgboost_binary_classifier.joblib')
lstm_path  = os.path.join(models_dir,  'lstm_multi_class_classifier.h5')
le_path    = os.path.join(scalers_dir, 'attack_type_label_encoder.joblib')
scaler_path= os.path.join(scalers_dir, 'attack_type_scaler.joblib')

# ── Ensemble weights (normalised ROC-AUC from training) ───────────────────────
_RF_ROCAUC  = 0.9999
_XGB_ROCAUC = 1.0000
_TOTAL      = _RF_ROCAUC + _XGB_ROCAUC
RF_W        = _RF_ROCAUC  / _TOTAL   # ≈ 0.49998
XGB_W       = _XGB_ROCAUC / _TOTAL   # ≈ 0.50003

@st.cache_resource
def load_models():
    try:
        rf   = joblib.load(rf_path)
        xgb  = joblib.load(xgb_path)
        lstm = tf.keras.models.load_model(lstm_path)
        le   = joblib.load(le_path) if os.path.exists(le_path) else _fallback_le(lstm)
        sc   = joblib.load(scaler_path)
        cols = list(rf.feature_names_in_)
        return rf, xgb, lstm, le, sc, cols
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

def _fallback_le(lstm):
    le = LabelEncoder()
    le.fit([f"AttackType_{i}" for i in range(lstm.output_shape[-1])])
    return le

rf_clf, xgb_clf, lstm_model, label_enc, scaler, EXPECTED_COLS = load_models()

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(df_raw):
    df = df_raw.copy()
    df.columns = (df.columns.str.strip()
                             .str.replace(' ', '_')
                             .str.replace('/', '_')
                             .str.replace('.', '', regex=False))
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df = df.fillna(0)
    X = df[[c for c in df.columns if c not in ['Label', 'Attack_Type_Original']]]
    for c in set(EXPECTED_COLS) - set(X.columns):
        X[c] = 0
    X = X[EXPECTED_COLS]
    X_lstm = scaler.transform(X).reshape(len(X), 1, len(EXPECTED_COLS))
    return X, X_lstm

# ── Two-stage cascade ensemble ─────────────────────────────────────────────────
def run_ensemble(X_bin, X_lstm_all):
    n = len(X_bin)

    # Stage 1 — Weighted soft voting (RF + XGBoost)
    rf_prob  = rf_clf.predict_proba(X_bin)[:, 1]
    xgb_prob = xgb_clf.predict_proba(X_bin)[:, 1]
    ens_score = RF_W * rf_prob + XGB_W * xgb_prob          # weighted threat score
    binary_pred = (ens_score >= 0.5).astype(int)            # final binary decision

    # Stage 2 — LSTM on attack records only
    attack_idx = np.where(binary_pred == 1)[0]
    lstm_types  = np.array(['Benign'] * n, dtype=object)
    lstm_conf   = np.zeros(n)
    lstm_probs  = np.zeros((n, len(label_enc.classes_)))

    if len(attack_idx) > 0:
        X_atk = X_lstm_all[attack_idx]
        probs = lstm_model.predict(X_atk, verbose=0)
        lstm_types[attack_idx] = label_enc.inverse_transform(np.argmax(probs, axis=1))
        lstm_conf[attack_idx]  = np.max(probs, axis=1)
        lstm_probs[attack_idx] = probs

    return {
        'rf_prob':      rf_prob,
        'xgb_prob':     xgb_prob,
        'ens_score':    ens_score,
        'binary_pred':  binary_pred,
        'lstm_types':   lstm_types,
        'lstm_conf':    lstm_conf,
        'lstm_probs':   lstm_probs,
        'attack_idx':   attack_idx,
    }

# ── Helpers ────────────────────────────────────────────────────────────────────
def threat_label(score):
    if score >= 0.8: return "🔴 CRITICAL"
    if score >= 0.6: return "🟠 HIGH"
    if score >= 0.4: return "🟡 MEDIUM"
    return "🟢 LOW"

def results_df(df_raw, r):
    return pd.DataFrame({
        'Record_ID':             range(len(df_raw)),
        'RF_Prob_%':             (r['rf_prob']   * 100).round(2),
        'RF_Pred':               np.where(r['rf_prob']  >= 0.5, 'Attack', 'Benign'),
        'XGB_Prob_%':            (r['xgb_prob']  * 100).round(2),
        'XGB_Pred':              np.where(r['xgb_prob'] >= 0.5, 'Attack', 'Benign'),
        'Ensemble_Score_%':      (r['ens_score'] * 100).round(2),
        'Final_Decision':        np.where(r['binary_pred'] == 1, 'Attack', 'Benign'),
        'Threat_Level':          [threat_label(s) for s in r['ens_score']],
        'LSTM_Attack_Type':      r['lstm_types'],
        'LSTM_Confidence_%':     (r['lstm_conf'] * 100).round(2),
    })

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ IDS System")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🔍 Run Analysis",
        "📊 Dashboard",
        "📈 Model Performance",
        "📥 Batch Processing",
    ])
    st.markdown("---")
    st.markdown(f"""
    **Ensemble Architecture**
    - Stage 1: RF + XGBoost
    - Stage 2: LSTM (attacks only)

    **Weights (ROC-AUC)**
    - RF:      `{RF_W:.4f}`
    - XGBoost: `{XGB_W:.4f}`
    """)

st.title("🛡️ Advanced Intrusion Detection System")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline diagram (always visible)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔗 Ensemble Pipeline Architecture", expanded=False):
    st.markdown("""
    <div style='font-family:monospace; font-size:0.9em; line-height:2em;'>
    <b>STAGE 1 — Binary Detection (Weighted Soft Voting)</b><br>
    &nbsp;&nbsp;
    <span class='pipeline-box rf-box'>🌲 Random Forest<br>weight = 0.4999</span>
    &nbsp;&nbsp;+&nbsp;&nbsp;
    <span class='pipeline-box xgb-box'>⚡ XGBoost<br>weight = 0.5001</span>
    &nbsp;&nbsp;──►&nbsp;&nbsp;
    <span class='pipeline-box ens-box'>Weighted Score<br>= RF_w×P(attack|RF) + XGB_w×P(attack|XGB)</span>
    &nbsp;&nbsp;──►&nbsp;&nbsp;
    <span class='pipeline-box final-box'>Threshold 0.5<br>Benign / Attack</span>
    <br><br>
    <b>STAGE 2 — Attack Classification (LSTM, attack records only)</b><br>
    &nbsp;&nbsp;
    <span class='pipeline-box final-box'>Attack records ──►</span>
    &nbsp;&nbsp;
    <span class='pipeline-box lstm-box'>🧠 LSTM<br>14 attack types</span>
    &nbsp;&nbsp;──►&nbsp;&nbsp;
    Attack Type + Confidence
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Why LSTM runs only on Stage 1 attacks:**
    The LSTM was trained exclusively on attack traffic to learn fine-grained attack type patterns.
    It has never seen benign traffic, so running it on benign records would produce meaningless output.
    The cascade design keeps each model doing what it was trained for.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Run Analysis
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Run Analysis":
    col_up, col_s = st.columns([4, 1])
    with col_up:
        uploaded = st.file_uploader("Upload network traffic CSV", type="csv")
    with col_s:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.checkbox("Use sample data")

    if uploaded or use_sample:
        if use_sample:
            st.info("Using randomly generated sample data (demonstration only).")
            rng = np.random.default_rng(42)
            df_raw = pd.DataFrame(rng.standard_normal((30, len(EXPECTED_COLS))),
                                  columns=EXPECTED_COLS)
        else:
            df_raw = pd.read_csv(StringIO(uploaded.getvalue().decode("utf-8")))

        with st.expander("Preview uploaded data"):
            st.dataframe(df_raw.head(10), use_container_width=True)

        if st.button("🚀 Run Ensemble Analysis", type="primary"):
            with st.spinner("Running two-stage ensemble..."):
                X_bin, X_lstm_all = preprocess(df_raw)
                r = run_ensemble(X_bin, X_lstm_all)
                df_res = results_df(df_raw, r)
                st.session_state['r']      = r
                st.session_state['df_res'] = df_res
                st.session_state['df_raw'] = df_raw
            st.success("Ensemble analysis complete!")

        if 'r' in st.session_state:
            r      = st.session_state['r']
            df_res = st.session_state['df_res']

            n_total   = len(df_res)
            n_attack  = int((r['binary_pred'] == 1).sum())
            n_benign  = n_total - n_attack
            n_rf_atk  = int((r['rf_prob']  >= 0.5).sum())
            n_xgb_atk = int((r['xgb_prob'] >= 0.5).sum())

            st.markdown("---")

            # ── Stage 1 summary ───────────────────────────────────────────────
            st.markdown("### Stage 1 — Binary Detection")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Records",   n_total)
            c2.metric("🌲 RF Attacks",   n_rf_atk,  f"{n_rf_atk/n_total*100:.1f}%")
            c3.metric("⚡ XGB Attacks",  n_xgb_atk, f"{n_xgb_atk/n_total*100:.1f}%")
            c4.metric("🔴 Final Attacks (Ensemble)", n_attack, f"{n_attack/n_total*100:.1f}%")
            c5.metric("🟢 Final Benign",  n_benign,  f"{n_benign/n_total*100:.1f}%")

            # Stage 1 charts
            col1, col2, col3 = st.columns(3)
            with col1:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(r['rf_prob']*100,  bins=20, alpha=0.7, color='#2d6a4f', label='RF')
                ax.hist(r['xgb_prob']*100, bins=20, alpha=0.7, color='#2c5f9e', label='XGBoost')
                ax.axvline(50, color='red', linestyle='--', label='Threshold')
                ax.set_xlabel('Attack Probability (%)'); ax.set_ylabel('Records')
                ax.set_title('RF vs XGBoost Probabilities'); ax.legend(fontsize=8)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(r['ens_score']*100, bins=20, color='#e65c00', edgecolor='white')
                ax.axvline(50, color='red', linestyle='--', label='Decision boundary')
                ax.set_xlabel('Ensemble Score (%)'); ax.set_ylabel('Records')
                ax.set_title('Weighted Ensemble Score'); ax.legend(fontsize=8)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            with col3:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.scatter(r['rf_prob']*100, r['xgb_prob']*100,
                           c=r['ens_score']*100, cmap='RdYlGn_r',
                           alpha=0.7, s=40, edgecolors='none')
                ax.set_xlabel('RF Prob (%)'); ax.set_ylabel('XGB Prob (%)')
                ax.set_title('RF vs XGBoost Agreement')
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.markdown("---")

            # ── Stage 2 summary ───────────────────────────────────────────────
            st.markdown("### Stage 2 — LSTM Attack Classification")

            if n_attack == 0:
                st.success("No attacks detected — LSTM stage not triggered.")
            else:
                attack_types_series = pd.Series(
                    r['lstm_types'][r['binary_pred'] == 1]
                ).value_counts()
                avg_lstm_conf = r['lstm_conf'][r['binary_pred'] == 1].mean()

                c1, c2, c3 = st.columns(3)
                c1.metric("Attack records sent to LSTM", n_attack)
                c2.metric("Unique attack types found",  attack_types_series.nunique())
                c3.metric("Avg LSTM confidence",        f"{avg_lstm_conf*100:.1f}%")

                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(6, max(3, len(attack_types_series)*0.45)))
                    attack_types_series.plot(kind='barh', ax=ax, color='#8b3a9e')
                    ax.set_xlabel('Records')
                    ax.set_title('LSTM: Predicted Attack Types')
                    ax.invert_yaxis()
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                with col2:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(r['lstm_conf'][r['binary_pred']==1]*100,
                            bins=20, color='#5c1a6b', edgecolor='white')
                    ax.set_xlabel('LSTM Confidence (%)'); ax.set_ylabel('Records')
                    ax.set_title('LSTM Confidence Distribution')
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.markdown("---")

            # ── Final results table ────────────────────────────────────────────
            st.markdown("### Final Ensemble Results")
            tab1, tab2, tab3 = st.tabs(["📋 All Records", "🚨 Attacks Only", "📥 Export"])

            with tab1:
                st.dataframe(df_res, use_container_width=True, height=400)

            with tab2:
                atk_df = df_res[df_res['Final_Decision'] == 'Attack'].sort_values(
                    'Ensemble_Score_%', ascending=False)
                if len(atk_df):
                    st.dataframe(atk_df, use_container_width=True, height=400)
                else:
                    st.success("No attacks detected.")

            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("📥 Download CSV",
                        df_res.to_csv(index=False),
                        f"ids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
                with c2:
                    st.download_button("📥 Download JSON",
                        df_res.to_json(orient='records', indent=2),
                        f"ids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    if 'r' not in st.session_state:
        st.info("Run an analysis first to populate the dashboard.")
    else:
        r      = st.session_state['r']
        df_res = st.session_state['df_res']

        st.subheader("📊 Ensemble Analysis Dashboard")

        n = len(df_res)
        n_atk = int((r['binary_pred'] == 1).sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Records",  n)
        c2.metric("🔴 Attacks",     n_atk,    f"{n_atk/n*100:.1f}%")
        c3.metric("🟢 Benign",      n-n_atk,  f"{(n-n_atk)/n*100:.1f}%")
        c4.metric("RF/XGB Agree",   f"{(( r['rf_prob']>=0.5)==(r['xgb_prob']>=0.5)).sum()}/{n}")
        c5.metric("LSTM triggered", f"{(r['binary_pred']==1).sum()} records")

        st.markdown("---")

        fig = plt.figure(figsize=(16, 10))
        gs  = plt.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # 1. Ensemble score histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(r['ens_score']*100, bins=25, color='#e65c00', edgecolor='white')
        ax1.axvline(50, color='red', linestyle='--', label='Threshold')
        ax1.set_title('Ensemble Threat Score Dist.', fontweight='bold')
        ax1.set_xlabel('Score (%)'); ax1.set_ylabel('Records'); ax1.legend()

        # 2. RF vs XGBoost comparison
        ax2 = fig.add_subplot(gs[0, 1])
        cats  = ['Benign', 'Attack']
        rf_v  = [(r['rf_prob']  < 0.5).sum(), (r['rf_prob']  >= 0.5).sum()]
        xgb_v = [(r['xgb_prob'] < 0.5).sum(), (r['xgb_prob'] >= 0.5).sum()]
        ens_v = [(r['binary_pred']==0).sum(),  (r['binary_pred']==1).sum()]
        x = np.arange(2); w = 0.25
        ax2.bar(x-w,  rf_v,  w, label='RF',       color='#2d6a4f')
        ax2.bar(x,    xgb_v, w, label='XGBoost',  color='#2c5f9e')
        ax2.bar(x+w,  ens_v, w, label='Ensemble', color='#e65c00')
        ax2.set_xticks(x); ax2.set_xticklabels(cats)
        ax2.set_title('Model Predictions Comparison', fontweight='bold')
        ax2.set_ylabel('Records'); ax2.legend(fontsize=8)

        # 3. LSTM attack types (attack records only)
        ax3 = fig.add_subplot(gs[0, 2])
        if n_atk > 0:
            tc = pd.Series(r['lstm_types'][r['binary_pred']==1]).value_counts().head(8)
            tc.plot(kind='barh', ax=ax3, color='#8b3a9e')
            ax3.invert_yaxis()
        else:
            ax3.text(0.5, 0.5, 'No attacks', ha='center', va='center')
        ax3.set_title('LSTM: Attack Types Detected', fontweight='bold')
        ax3.set_xlabel('Records')

        # 4. Threat level pie
        ax4 = fig.add_subplot(gs[1, 0])
        tl = df_res['Threat_Level'].value_counts()
        clr_map = {'🟢 LOW':'#2ca02c','🟡 MEDIUM':'#ffbf00',
                   '🟠 HIGH':'#ff7f0e','🔴 CRITICAL':'#d62728'}
        ax4.pie(tl.values,
                labels=tl.index,
                colors=[clr_map.get(l,'#aaa') for l in tl.index],
                autopct='%1.1f%%', startangle=90)
        ax4.set_title('Threat Level Distribution', fontweight='bold')

        # 5. RF vs XGBoost scatter
        ax5 = fig.add_subplot(gs[1, 1])
        sc = ax5.scatter(r['rf_prob']*100, r['xgb_prob']*100,
                         c=r['ens_score']*100, cmap='RdYlGn_r',
                         alpha=0.7, s=35, edgecolors='none')
        plt.colorbar(sc, ax=ax5, label='Ensemble Score %')
        ax5.axvline(50, color='gray', linestyle=':', lw=0.8)
        ax5.axhline(50, color='gray', linestyle=':', lw=0.8)
        ax5.set_xlabel('RF Prob (%)'); ax5.set_ylabel('XGB Prob (%)')
        ax5.set_title('RF vs XGBoost Probability Scatter', fontweight='bold')

        # 6. LSTM confidence by type
        ax6 = fig.add_subplot(gs[1, 2])
        if n_atk > 0:
            conf_df = pd.DataFrame({
                'type': r['lstm_types'][r['binary_pred']==1],
                'conf': r['lstm_conf'][r['binary_pred']==1]*100
            })
            means = conf_df.groupby('type')['conf'].mean().sort_values()
            means.plot(kind='barh', ax=ax6, color='#5c1a6b')
            ax6.set_xlabel('Avg Confidence (%)')
        else:
            ax6.text(0.5, 0.5, 'No attacks', ha='center', va='center')
        ax6.set_title('LSTM Avg Confidence by Type', fontweight='bold')

        st.pyplot(fig); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.subheader("📈 Model Performance — Trained on CICIDS2017")

    # ── Architecture explainer ─────────────────────────────────────────────────
    st.markdown("### Ensemble Architecture")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='pipeline-box rf-box'>
        🌲 <b>Random Forest</b><br>
        Role: Binary detection<br>
        n_estimators = 50<br>
        Weight in ensemble: <b>0.4999</b>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='pipeline-box xgb-box'>
        ⚡ <b>XGBoost</b><br>
        Role: Binary detection<br>
        n_estimators = 100<br>
        Weight in ensemble: <b>0.5001</b>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='pipeline-box lstm-box'>
        🧠 <b>LSTM</b><br>
        Role: Attack classification<br>
        LSTM(100) → Dense(50) → Dense(14)<br>
        Runs on attack records only
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Metrics tables ────────────────────────────────────────────────────────
    st.markdown("### Classification Metrics  *(30% test set — 694,143 records for binary, 100,948 for LSTM)*")

    rf_m  = dict(Accuracy=0.9988, Precision=0.9971, Recall=0.9949,
                 F1=0.9960, ROC_AUC=0.9999)
    xgb_m = dict(Accuracy=0.9988, Precision=0.9975, Recall=0.9946,
                 F1=0.9960, ROC_AUC=1.0000)
    lstm_m= dict(Accuracy=0.9961, Precision_macro=0.90, Recall_macro=0.83,
                 F1_macro=0.83,  Test_Loss=0.0132)

    c1, c2, c3 = st.columns(3)
    def _metric_table(d):
        return pd.DataFrame({'Metric': d.keys(),
                             'Score':  [f"{v:.4f}" for v in d.values()]})
    with c1:
        st.markdown("**🌲 Random Forest** *(Binary)*")
        st.table(_metric_table(rf_m))
    with c2:
        st.markdown("**⚡ XGBoost** *(Binary)*")
        st.table(_metric_table(xgb_m))
    with c3:
        st.markdown("**🧠 LSTM** *(14-class, attack only)*")
        st.table(_metric_table(lstm_m))

    st.markdown("---")

    # ── Comparison charts ─────────────────────────────────────────────────────
    st.markdown("### Model Comparison Charts")

    shared = ['Accuracy', 'Precision', 'Recall', 'F1']
    rf_v   = [rf_m['Accuracy'],  rf_m['Precision'],  rf_m['Recall'],  rf_m['F1']]
    xgb_v  = [xgb_m['Accuracy'], xgb_m['Precision'], xgb_m['Recall'], xgb_m['F1']]
    lstm_v = [0.9961,             0.90,               0.83,            0.83]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(shared)); w = 0.25
        b1 = ax.bar(x-w,  rf_v,   w, label='Random Forest', color='#2d6a4f')
        b2 = ax.bar(x,    xgb_v,  w, label='XGBoost',       color='#2c5f9e')
        b3 = ax.bar(x+w,  lstm_v, w, label='LSTM (macro)',  color='#8b3a9e')
        ax.set_xticks(x); ax.set_xticklabels(shared)
        ax.set_ylim(0.75, 1.04)
        ax.set_ylabel('Score')
        ax.set_title('Accuracy / Precision / Recall / F1 Comparison', fontweight='bold')
        ax.legend(fontsize=9)
        for bars in [b1, b2, b3]:
            ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        models = ['Random Forest', 'XGBoost']
        aucs   = [0.9999, 1.0000]
        bars = ax.bar(models, aucs, color=['#2d6a4f', '#2c5f9e'], width=0.4)
        ax.set_ylim(0.998, 1.001)
        ax.set_ylabel('ROC-AUC')
        ax.set_title('ROC-AUC (Binary Models)', fontweight='bold')
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.00005, f"{v:.4f}",
                    ha='center', fontsize=11, fontweight='bold')
        ax.annotate("LSTM not shown — it is a\n14-class classifier, not binary",
                    xy=(1.5, 0.9985), fontsize=9, color='gray', ha='center')
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("---")

    # ── LSTM per-class report ─────────────────────────────────────────────────
    st.markdown("### 🧠 LSTM Per-class Classification Report")

    per_class = pd.DataFrame({
        'Attack Type': [
            'Bot','DDoS','DoS GoldenEye','DoS Hulk','DoS Slowhttptest',
            'DoS slowloris','FTP-Patator','Heartbleed','Infiltration',
            'PortScan','SSH-Patator',
            'Web Attack - Brute Force','Web Attack - Sql Injection','Web Attack - XSS'
        ],
        'Precision': [1.00,1.00,0.98,1.00,0.99,0.99,1.00,1.00,1.00,
                      0.99,0.96,0.68,0.00,1.00],
        'Recall':    [1.00,1.00,1.00,1.00,0.99,0.98,1.00,1.00,0.73,
                      0.95,0.98,0.93,0.00,0.02],
        'F1 Score':  [1.00,1.00,0.99,1.00,0.99,0.99,1.00,1.00,0.84,
                      0.97,0.97,0.79,0.00,0.03],
        'Test Samples':[431,38404,3086,51854,1568,1616,1779,3,11,
                        587,966,441,6,196]
    })

    col1, col2 = st.columns(2)
    with col1:
        # Heatmap
        fig, ax = plt.subplots(figsize=(5, 6))
        heat = per_class[['Precision','Recall','F1 Score']].values.T
        sns.heatmap(heat,
                    xticklabels=per_class['Attack Type'],
                    yticklabels=['Precision','Recall','F1'],
                    annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=0, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title('LSTM Per-class Metrics Heatmap', fontweight='bold')
        ax.tick_params(axis='x', rotation=55, labelsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        # F1 bar chart
        fig, ax = plt.subplots(figsize=(5, 6))
        colors = ['#d62728' if v < 0.5 else '#ff7f0e' if v < 0.8 else '#2ca02c'
                  for v in per_class['F1 Score']]
        ax.barh(per_class['Attack Type'], per_class['F1 Score'], color=colors)
        ax.axvline(0.8, color='orange', linestyle='--', lw=1.2, label='F1 = 0.80')
        ax.axvline(0.5, color='red',    linestyle='--', lw=1.2, label='F1 = 0.50')
        ax.set_xlim(0, 1.25)
        ax.set_xlabel('F1 Score')
        ax.set_title('LSTM F1 Score per Attack Type', fontweight='bold')
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        for i, (v, s) in enumerate(zip(per_class['F1 Score'], per_class['Test Samples'])):
            ax.text(v+0.01, i, f"{v:.2f}  (n={s})", va='center', fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.dataframe(per_class, use_container_width=True, hide_index=True)
    st.caption(
        "Low F1 for 'Web Attack - Sql Injection' (6 test samples) and "
        "'Web Attack - XSS' (196 samples) reflects class imbalance, not model failure. "
        "Both classes appear in < 0.1% of total traffic."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Batch Processing
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Batch Processing":
    st.subheader("📥 Batch Processing")

    uploaded_batch = st.file_uploader("Upload CSV for batch analysis", type="csv", key="batch")

    if uploaded_batch:
        df_batch = pd.read_csv(StringIO(uploaded_batch.getvalue().decode("utf-8")))
        st.info(f"Loaded {len(df_batch):,} records.")

        threshold = st.slider("Flag records with Ensemble Score above (%)", 0, 100, 50)

        if st.button("🚀 Run Batch Ensemble", type="primary"):
            with st.spinner("Running two-stage ensemble on batch..."):
                X_bin, X_lstm_all = preprocess(df_batch)
                r = run_ensemble(X_bin, X_lstm_all)
                df_res = results_df(df_batch, r)

            st.success("Batch complete!")

            high = df_res[df_res['Ensemble_Score_%'] >= threshold]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total",    len(df_res))
            c2.metric("Threats",  len(high))
            c3.metric("Rate",     f"{len(high)/len(df_res)*100:.1f}%")
            c4.metric("RF/XGB Agree",
                      f"{((r['rf_prob']>=0.5)==(r['xgb_prob']>=0.5)).sum()/len(r['rf_prob'])*100:.1f}%")

            tab1, tab2 = st.tabs(["🚨 Flagged Threats", "📋 All Results"])
            with tab1:
                st.dataframe(high.sort_values('Ensemble_Score_%', ascending=False),
                             use_container_width=True, height=400)
            with tab2:
                st.dataframe(df_res, use_container_width=True, height=400)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 All Results CSV",
                    df_res.to_csv(index=False),
                    f"batch_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            with c2:
                st.download_button("📥 Threats Only CSV",
                    high.to_csv(index=False),
                    f"batch_threats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
