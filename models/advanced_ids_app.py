"""
Advanced Intrusion Detection System (IDS) - Professional UI
Built with Streamlit for real-time threat detection and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Advanced IDS System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .threat-critical {
        background: #FF6B6B;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .threat-high {
        background: #FFA500;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .threat-medium {
        background: #FFD700;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .threat-low {
        background: #90EE90;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .info-box {
        background: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #1976D2;
    }
    .success-box {
        background: #E8F5E9;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #388E3C;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL PATHS & CONFIGURATION
# ============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = base_dir
scalers_dir = os.path.join(os.path.dirname(base_dir), 'scalers')

rf_model_path = os.path.join(output_dir, 'random_forest_binary_classifier.joblib')
xgb_model_path = os.path.join(output_dir, 'xgboost_binary_classifier.joblib')
lstm_model_path = os.path.join(output_dir, 'lstm_multi_class_classifier.h5')
label_encoder_path = os.path.join(scalers_dir, 'attack_type_label_encoder.joblib')
scaler_path = os.path.join(scalers_dir, 'attack_type_scaler.joblib')

# ============================================================================
# LOAD MODELS & RESOURCES
# ============================================================================
@st.cache_resource
def load_resources():
    """Load all ML models and preprocessors with error handling"""
    try:
        rf_classifier = joblib.load(rf_model_path)
        xgb_classifier = joblib.load(xgb_model_path)
        lstm_model = tf.keras.models.load_model(lstm_model_path)

        if os.path.exists(label_encoder_path):
            label_encoder = joblib.load(label_encoder_path)
        else:
            fallback_classes = [f"AttackType_{i}" for i in range(lstm_model.output_shape[-1] if lstm_model.output_shape is not None else 1)]
            label_encoder = LabelEncoder()
            label_encoder.fit(fallback_classes)
            st.warning("⚠️ Attack type label encoder not found. Using fallback attack labels.")

        scaler = joblib.load(scaler_path)
        expected_columns = list(rf_classifier.feature_names_in_)

        return rf_classifier, xgb_classifier, lstm_model, label_encoder, scaler, expected_columns
    except Exception as e:
        st.error(f"❌ Error loading resources: {e}\nPlease ensure all models are saved in {output_dir} and the scaler exists in {scalers_dir}")
        st.stop()

rf_classifier, xgb_classifier, lstm_model, label_encoder, scaler, expected_columns = load_resources()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def preprocess_input_data(df_raw):
    """Clean and preprocess input data to match training pipeline"""
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.replace('.', '', regex=False)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df = df.fillna(0)
    
    X_binary_cols = [col for col in df.columns if col not in ['Label', 'Attack_Type_Original']]
    X_binary = df[X_binary_cols]
    
    missing_cols_binary = set(expected_columns) - set(X_binary.columns)
    for c in missing_cols_binary:
        X_binary[c] = 0
    X_binary = X_binary[expected_columns]
    
    X_multiclass = X_binary.copy()
    X_multiclass_scaled = scaler.transform(X_multiclass)
    num_features = X_multiclass_scaled.shape[1]
    X_multiclass_reshaped = X_multiclass_scaled.reshape(X_multiclass_scaled.shape[0], 1, num_features)
    
    return X_binary, X_multiclass_reshaped

def get_threat_level(prob):
    """Convert probability to threat level emoji"""
    if prob >= 0.8:
        return "🔴 CRITICAL"
    elif prob >= 0.6:
        return "🟠 HIGH"
    elif prob >= 0.4:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"

def get_threat_color(prob):
    """Get CSS class based on threat level"""
    if prob >= 0.8:
        return "threat-critical"
    elif prob >= 0.6:
        return "threat-high"
    elif prob >= 0.4:
        return "threat-medium"
    else:
        return "threat-low"

def create_threat_gauge(threat_score):
    """Create a gauge chart for threat visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=threat_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Threat Score"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#90EE90"},
                {'range': [40, 60], 'color': "#FFD700"},
                {'range': [60, 80], 'color': "#FFA500"},
                {'range': [80, 100], 'color': "#FF6B6B"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

# ============================================================================
# MAIN HEADER
# ============================================================================
st.markdown('<h1 class="main-header">🛡️ Advanced Intrusion Detection System</h1>', unsafe_allow_html=True)
st.markdown('*Real-time Network Threat Detection using Ensemble ML Models*')
st.markdown('---')

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.title("⚙️ System Navigation")
    st.markdown("---")
    
    system_mode = st.radio(
        "Select Module:",
        [
            "🚀 Real-time Analysis",
            "📊 Dashboard Analytics",
            "⚖️ Model Comparison",
            "ℹ️ System Information",
            "📥 Batch Processing",
            "📋 History"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    if st.session_state.predictions is not None:
        total_records = len(st.session_state.predictions)
        threats = (st.session_state.predictions['Ensemble_Threat_Score'] >= 50).sum()
        st.metric("Records Analyzed", total_records)
        st.metric("Threats Detected", threats)
        st.metric("Threat Rate", f"{threats/total_records*100:.1f}%" if total_records > 0 else "0%")

# ============================================================================
# MAIN CONTENT - MODE SELECTION
# ============================================================================

if system_mode == "🚀 Real-time Analysis":
    st.subheader("📤 Upload & Analyze Network Traffic")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        uploaded_file = st.file_uploader("📁 Choose a CSV file containing network traffic data", type="csv")
    with col2:
        st.markdown("")
        st.markdown("")
        use_sample = st.checkbox("📊 Use Sample Data")
    with col3:
        st.markdown("")
        st.markdown("")
        confidence_threshold = st.slider("Threshold", 0, 100, 50, key="threshold1")
    
    if uploaded_file is not None or use_sample:
        if use_sample:
            st.info("📌 Using synthetic sample data for demonstration")
            df_raw = pd.DataFrame(np.random.randn(10, 80), columns=[f'feature_{i}' for i in range(80)])
        else:
            data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df_raw = pd.read_csv(data)
        
        st.session_state.raw_data = df_raw
        
        # Data Preview Section
        with st.expander("📋 Data Preview", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Records", len(df_raw))
            with col2:
                st.metric("🔧 Features", df_raw.shape[1])
            with col3:
                st.metric("💾 Total Points", len(df_raw) * df_raw.shape[1])
            with col4:
                st.metric("📌 Memory (est.)", f"{df_raw.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            st.dataframe(df_raw.head(), use_container_width=True)
        
        st.markdown("---")
        
        # Analysis Button
        if st.button("🔍 Run Threat Analysis", type="primary"):
            # Progress Tracking
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            try:
                # Step 1: Preprocessing
                status_placeholder.info("🔄 Step 1/4: Preprocessing data...")
                progress_bar.progress(15)
                X_binary, X_multiclass_reshaped = preprocess_input_data(df_raw)
                
                # Step 2: Random Forest
                status_placeholder.info("🔄 Step 2/4: Running Random Forest classifier...")
                progress_bar.progress(40)
                binary_pred_rf = rf_classifier.predict(X_binary)
                binary_proba_rf = rf_classifier.predict_proba(X_binary)[:, 1]
                
                # Step 3: XGBoost
                status_placeholder.info("🔄 Step 3/4: Running XGBoost classifier...")
                progress_bar.progress(65)
                binary_pred_xgb = xgb_classifier.predict(X_binary)
                binary_proba_xgb = xgb_classifier.predict_proba(X_binary)[:, 1]
                
                # Step 4: LSTM
                status_placeholder.info("🔄 Step 4/4: Running LSTM attack classification...")
                progress_bar.progress(85)
                multi_pred_proba_lstm = lstm_model.predict(X_multiclass_reshaped, verbose=0)
                multi_pred_lstm_encoded = np.argmax(multi_pred_proba_lstm, axis=1)
                multi_pred_lstm_labels = label_encoder.inverse_transform(multi_pred_lstm_encoded)
                
                progress_bar.progress(100)
                status_placeholder.success("✅ Analysis complete!")
                
                # Ensemble Voting
                ensemble_threat = (binary_pred_rf + binary_pred_xgb) / 2.0
                
                # Build Results DataFrame
                results_df = pd.DataFrame({
                    'Record_ID': range(len(df_raw)),
                    'RF_Prediction': np.where(binary_pred_rf == 1, '🔴 Attack', '🟢 Benign'),
                    'RF_Confidence': (np.maximum(binary_proba_rf, 1 - binary_proba_rf) * 100).round(2),
                    'XGB_Prediction': np.where(binary_pred_xgb == 1, '🔴 Attack', '🟢 Benign'),
                    'XGB_Confidence': (np.maximum(binary_proba_xgb, 1 - binary_proba_xgb) * 100).round(2),
                    'Ensemble_Threat_Score': (ensemble_threat * 100).round(2),
                    'Attack_Type': multi_pred_lstm_labels,
                    'Attack_Confidence': (np.max(multi_pred_proba_lstm, axis=1) * 100).round(2),
                    'Threat_Level': [get_threat_level(x) for x in ensemble_threat]
                })
                
                st.session_state.predictions = results_df
                st.session_state.processing_history.append({
                    'timestamp': datetime.now(),
                    'records': len(df_raw),
                    'threats': (ensemble_threat >= 0.5).sum()
                })
                
                st.markdown("---")
                
                # Threat Assessment Summary
                st.subheader("🎯 Threat Assessment Summary")
                
                attacks = (ensemble_threat >= 0.5).sum()
                benign = len(results_df) - attacks
                critical = (ensemble_threat >= 0.8).sum()
                high_risk = ((ensemble_threat >= 0.6) & (ensemble_threat < 0.8)).sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔴 Attacks Detected", attacks, f"{attacks/len(results_df)*100:.1f}%")
                with col2:
                    st.metric("🟢 Benign Traffic", benign)
                with col3:
                    st.metric("🔴 Critical Threats", critical)
                with col4:
                    st.metric("🟠 High Risk", high_risk)
                
                st.markdown("---")
                
                # Results Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Full Results", "⚠️ High Threats", "📈 Analytics", "📥 Export"])
                
                with tab1:
                    st.dataframe(results_df, use_container_width=True, height=400)
                
                with tab2:
                    threats_df = results_df[results_df['Ensemble_Threat_Score'] >= confidence_threshold].sort_values('Ensemble_Threat_Score', ascending=False)
                    if len(threats_df) > 0:
                        st.warning(f"⚠️ Found {len(threats_df)} records exceeding {confidence_threshold}% threat threshold")
                        st.dataframe(threats_df, use_container_width=True, height=400)
                    else:
                        st.success("✅ No threats detected above threshold!")
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Model Consensus Analysis**")
                        agreement = (binary_pred_rf == binary_pred_xgb).sum() / len(binary_pred_rf) * 100
                        st.metric("Model Agreement Rate", f"{agreement:.1f}%")
                        
                        # Threat Distribution
                        threat_counts = {
                            '🟢 Low': (ensemble_threat < 0.4).sum(),
                            '🟡 Medium': ((ensemble_threat >= 0.4) & (ensemble_threat < 0.6)).sum(),
                            '🟠 High': ((ensemble_threat >= 0.6) & (ensemble_threat < 0.8)).sum(),
                            '🔴 Critical': (ensemble_threat >= 0.8).sum()
                        }
                        st.dataframe(pd.DataFrame(list(threat_counts.items()), columns=['Level', 'Count']))
                    
                    with col2:
                        # Threat Distribution Chart
                        fig = go.Figure(data=[go.Pie(labels=list(threat_counts.keys()), values=list(threat_counts.values()))])
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Confidence Distribution
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.hist([results_df['RF_Confidence'], results_df['XGB_Confidence']], 
                             label=['Random Forest', 'XGBoost'], bins=20, alpha=0.7, color=['#667eea', '#764ba2'])
                    ax1.set_title('Model Confidence Distribution', fontweight='bold')
                    ax1.set_xlabel('Confidence (%)')
                    ax1.set_ylabel('Frequency')
                    ax1.legend()
                    
                    # Attack Type Distribution (if attacks present)
                    if attacks > 0:
                        attack_types = results_df[results_df['Ensemble_Threat_Score'] > 50]['Attack_Type'].value_counts()
                        attack_types.plot(kind='barh', ax=ax2, color='#FF6B6B')
                        ax2.set_title('Detected Attack Types', fontweight='bold')
                        ax2.set_xlabel('Count')
                    
                    st.pyplot(fig)
                
                with tab4:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full CSV",
                            data=csv_data,
                            file_name=f"ids_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        threats_only = results_df[results_df['Ensemble_Threat_Score'] >= confidence_threshold]
                        if len(threats_only) > 0:
                            csv_threats = threats_only.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Threats CSV",
                                data=csv_threats,
                                file_name=f"ids_threats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    with col3:
                        json_data = results_df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=f"ids_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    # Export Summary Report
                    st.write("**Summary Statistics**")
                    summary = pd.DataFrame({
                        'Metric': ['Total Records', 'Attacks', 'Benign', 'Critical Threats', 'Avg Threat Score'],
                        'Value': [
                            len(results_df),
                            attacks,
                            benign,
                            critical,
                            f"{ensemble_threat.mean()*100:.2f}%"
                        ]
                    })
                    st.dataframe(summary, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                progress_bar.empty()
                status_placeholder.empty()

elif system_mode == "📊 Dashboard Analytics":
    st.subheader("📈 System Dashboard & Analytics")
    
    if st.session_state.predictions is not None:
        results_df = st.session_state.predictions
        
        # KPI Section
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Records Processed", len(results_df))
        with col2:
            attack_count = (results_df['Ensemble_Threat_Score'] >= 50).sum()
            st.metric("🔴 Threats", attack_count, f"{attack_count/len(results_df)*100:.1f}%")
        with col3:
            st.metric("🟢 Safe Traffic", len(results_df) - attack_count)
        with col4:
            avg_score = results_df['Ensemble_Threat_Score'].mean()
            st.metric("📊 Avg Score", f"{avg_score:.1f}%")
        
        st.markdown("---")
        
        # Advanced Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Threat Distribution
            fig = px.pie(
                values=results_df['Ensemble_Threat_Score'].value_counts().values[:5],
                names=['Level ' + str(i) for i in range(5)],
                title="Threat Level Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Ensemble Threat Score Distribution
            fig = px.histogram(
                results_df,
                x='Ensemble_Threat_Score',
                nbins=30,
                title="Threat Score Distribution",
                labels={'Ensemble_Threat_Score': 'Threat Score (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model Comparison
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Model Predictions**")
            rf_counts = results_df['RF_Prediction'].value_counts()
            fig = px.bar(x=rf_counts.index, y=rf_counts.values, title="Random Forest Predictions")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Model Predictions**")
            xgb_counts = results_df['XGB_Prediction'].value_counts()
            fig = px.bar(x=xgb_counts.index, y=xgb_counts.values, title="XGBoost Predictions")
            st.plotly_chart(fig, use_container_width=True)
        
        # Attack Types (if available)
        attack_mask = results_df['Ensemble_Threat_Score'] >= 50
        if attack_mask.any():
            st.write("**Top Detected Attack Types**")
            attack_types = results_df[attack_mask]['Attack_Type'].value_counts().head(10)
            fig = px.barh(y=attack_types.index, x=attack_types.values, title="Attack Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📌 No analysis results available. Please run analysis first from the 'Real-time Analysis' tab.")

elif system_mode == "⚖️ Model Comparison":
    st.subheader("⚖️ Model Performance Comparison")
    
    if st.session_state.predictions is not None:
        results_df = st.session_state.predictions
        
        st.info("💡 Comparing predictions from all three models: Random Forest, XGBoost, and LSTM")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🌲 Random Forest")
            rf_benign = (results_df['RF_Prediction'] == '🟢 Benign').sum()
            rf_attack = (results_df['RF_Prediction'] == '🔴 Attack').sum()
            st.metric("Benign", rf_benign)
            st.metric("Attack", rf_attack)
            st.metric("Avg Confidence", f"{results_df['RF_Confidence'].mean():.1f}%")
        
        with col2:
            st.markdown("### ⚡ XGBoost")
            xgb_benign = (results_df['XGB_Prediction'] == '🟢 Benign').sum()
            xgb_attack = (results_df['XGB_Prediction'] == '🔴 Attack').sum()
            st.metric("Benign", xgb_benign)
            st.metric("Attack", xgb_attack)
            st.metric("Avg Confidence", f"{results_df['XGB_Confidence'].mean():.1f}%")
        
        with col3:
            st.markdown("### 🧠 LSTM")
            st.metric("Attack Types", results_df['Attack_Type'].nunique())
            st.metric("Avg Confidence", f"{results_df['Attack_Confidence'].mean():.1f}%")
            st.metric("Records", len(results_df))
        
        st.markdown("---")
        
        # Model Agreement Analysis
        st.subheader("🤝 Model Agreement Analysis")
        agreement = (results_df['RF_Prediction'] == results_df['XGB_Prediction']).sum() / len(results_df) * 100
        st.progress(min(agreement / 100, 1.0))
        st.write(f"**Agreement Rate: {agreement:.1f}%**")
        
        # Detailed Comparison Table
        st.write("**Detailed Comparison**")
        comparison_df = results_df[['Record_ID', 'RF_Prediction', 'RF_Confidence', 
                                     'XGB_Prediction', 'XGB_Confidence', 'Attack_Type', 'Ensemble_Threat_Score']]
        st.dataframe(comparison_df, use_container_width=True, height=400)
    
    else:
        st.info("📌 No analysis results available. Please run analysis first.")

elif system_mode == "ℹ️ System Information":
    st.subheader("🤖 System Information & Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Model Architecture
        
        **Binary Classification:**
        - 🌲 **Random Forest**: Ensemble of 50 decision trees
        - ⚡ **XGBoost**: Gradient boosting classifier
        
        **Multi-class Classification:**
        - 🧠 **LSTM**: Deep neural network (Keras/TensorFlow)
        
        ### 🎯 Threat Level Definitions
        - 🟢 **Low**: < 40% probability
        - 🟡 **Medium**: 40-60% probability
        - 🟠 **High**: 60-80% probability
        - 🔴 **Critical**: > 80% probability
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Features & Capabilities
        
        **Input Features:**
        - ~80 network traffic metrics
        - Flow statistics
        - Protocol analysis
        - Temporal patterns
        
        **Output Metrics:**
        - Binary attack classification
        - Threat probability scores
        - Attack type identification
        - Ensemble consensus voting
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🔧 Technical Specifications
    
    **Dataset**: CICIDS2017 - Canadian Institute for Cybersecurity
    
    **Training Data**: Network traffic samples with labeled attack types
    
    **Models**: 3 complementary ML models providing ensemble predictions
    
    **Real-time Processing**: Support for batch and single-record analysis
    """)
    
    st.markdown("---")
    st.markdown("""
    ### ⚙️ System Architecture
    
    1. **Data Preprocessing**: Cleaning, normalization, feature alignment
    2. **Model Inference**: Parallel predictions from 3 models
    3. **Ensemble Voting**: Weighted consensus for final decision
    4. **Threat Scoring**: Probabilistic threat assessment
    5. **Reporting**: Detailed analysis and export capabilities
    """)

elif system_mode == "📥 Batch Processing":
    st.subheader("📦 Batch Processing & Automation")
    
    st.markdown("""
    ### 🚀 Batch Processing Features
    - Process multiple records simultaneously
    - Automated threat detection
    - Bulk export in multiple formats
    - Performance aggregation
    - Alert generation for high-risk scenarios
    """)
    
    uploaded_file = st.file_uploader("📁 Upload CSV for batch processing", type="csv", key="batch_upload")
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Records", len(df_batch))
        with col2:
            st.metric("🔧 Features", df_batch.shape[1])
        with col3:
            batch_size = st.slider("Batch Size", 10, 1000, 100)
        
        confidence_threshold = st.slider("Threat Threshold (%)", 0, 100, 50)
        
        if st.button("🚀 Start Batch Processing", type="primary"):
            with st.spinner("⏳ Processing batch data..."):
                X_binary, X_multiclass_reshaped = preprocess_input_data(df_batch)
                
                binary_pred_rf = rf_classifier.predict(X_binary)
                binary_proba_rf = rf_classifier.predict_proba(X_binary)[:, 1]
                
                binary_pred_xgb = xgb_classifier.predict(X_binary)
                binary_proba_xgb = xgb_classifier.predict_proba(X_binary)[:, 1]
                
                multi_pred_proba_lstm = lstm_model.predict(X_multiclass_reshaped, verbose=0)
                multi_pred_lstm_encoded = np.argmax(multi_pred_proba_lstm, axis=1)
                multi_pred_lstm_labels = label_encoder.inverse_transform(multi_pred_lstm_encoded)
                
                ensemble_threat = (binary_pred_rf + binary_pred_xgb) / 2.0
                
                batch_results = pd.DataFrame({
                    'Record_ID': range(len(df_batch)),
                    'RF_Prediction': np.where(binary_pred_rf == 1, 'Attack', 'Benign'),
                    'XGB_Prediction': np.where(binary_pred_xgb == 1, 'Attack', 'Benign'),
                    'Ensemble_Threat_Score': ensemble_threat * 100,
                    'Attack_Type': multi_pred_lstm_labels,
                    'Threat_Level': [get_threat_level(x) for x in ensemble_threat]
                })
                
                high_threat = batch_results[batch_results['Ensemble_Threat_Score'] >= confidence_threshold]
                
                st.success(f"✅ Batch processing complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Records", len(batch_results))
                with col2:
                    st.metric("⚠️ Threats", len(high_threat))
                with col3:
                    st.metric("📈 Threat Rate", f"{len(high_threat)/len(batch_results)*100:.1f}%")
                
                st.markdown("---")
                
                if len(high_threat) > 0:
                    st.warning(f"🚨 Found {len(high_threat)} high-threat records!")
                    st.dataframe(high_threat.sort_values('Ensemble_Threat_Score', ascending=False), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = batch_results.to_csv(index=False)
                        st.download_button("📥 Full Results", csv, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    with col2:
                        threats_csv = high_threat.to_csv(index=False)
                        st.download_button("📥 Threats Only", threats_csv, f"threats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                else:
                    st.success("✅ No high-threat records detected!")

elif system_mode == "📋 History":
    st.subheader("📋 Analysis History")
    
    if len(st.session_state.processing_history) > 0:
        history_df = pd.DataFrame(st.session_state.processing_history)
        st.dataframe(history_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Analyses", len(history_df))
        with col2:
            st.metric("Total Records", history_df['records'].sum())
        
        # Summary Chart
        fig = px.bar(history_df, x='timestamp', y=['records', 'threats'], title="Analysis History")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📌 No analysis history yet. Run an analysis to see history.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>🛡️ Advanced IDS System | Powered by Machine Learning</div>", unsafe_allow_html=True)
