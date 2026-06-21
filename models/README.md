# 🛡️ Advanced Intrusion Detection System (IDS)

A professional-grade, real-time network intrusion detection system powered by machine learning ensemble models. This system uses Random Forest, XGBoost, and LSTM neural networks to detect and classify network attacks with high accuracy.

## 🌟 Features

### Core Capabilities
- ✅ **Real-time Threat Detection**: Instant classification of network traffic
- ✅ **Multi-Model Ensemble**: Combined predictions from 3 complementary ML models
- ✅ **Attack Type Classification**: Identify specific attack categories
- ✅ **Threat Severity Scoring**: 4-level threat classification system
- ✅ **Batch Processing**: Analyze 1000s of records efficiently
- ✅ **Professional Dashboard**: Rich visualizations and analytics
- ✅ **Multiple Export Formats**: CSV, JSON export capabilities
- ✅ **Model Comparison**: Detailed analysis of individual model performance

### Technical Features
- Machine Learning Models:
  - 🌲 Random Forest (Binary Classification)
  - ⚡ XGBoost (Binary Classification)
  - 🧠 LSTM Neural Network (Multi-class Attack Type Detection)
- Ensemble Voting System
- Real-time Data Preprocessing
- Comprehensive Error Handling
- Session-based State Management

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run advanced_ids_app.py
```

### 3. Or Use the Setup Script
```bash
python setup_ids.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 Application Modules

### 🚀 Real-time Analysis
- Upload CSV files with network traffic data
- View live threat predictions
- Detailed per-record analysis
- Threat level assignment
- Export results

### 📊 Dashboard Analytics
- Comprehensive visualizations
- Threat distribution charts
- Model comparison plots
- Attack type analysis
- System-wide statistics

### ⚖️ Model Comparison
- Individual model metrics
- Model agreement analysis
- Confidence distribution
- Performance comparison
- Detailed comparison tables

### ℹ️ System Information
- Model architecture details
- Threat level definitions
- Technical specifications
- System components

### 📥 Batch Processing
- Bulk upload and processing
- Configurable threat threshold
- High-threat record filtering
- Batch export functionality

### 📋 Analysis History
- Track analysis activities
- Historical statistics
- Trend visualization

## 📈 Threat Classification

| Level | Score | Color | Action |
|-------|-------|-------|--------|
| 🟢 LOW | < 40% | Green | Monitor |
| 🟡 MEDIUM | 40-60% | Yellow | Review |
| 🟠 HIGH | 60-80% | Orange | Alert |
| 🔴 CRITICAL | > 80% | Red | Immediate Action |

## 🤖 Model Architecture

### Binary Classification Models
**Purpose**: Detect if traffic is Attack or Benign

- **Random Forest**: 50-tree ensemble, fast and interpretable
- **XGBoost**: Gradient boosting for high accuracy

### Multi-class Classification Model
**Purpose**: Identify specific attack type

- **LSTM**: Deep neural network with 1 LSTM layer, 2 Dense layers

### Ensemble Strategy
- Majority voting for attack/benign decision
- Confidence-weighted scoring
- Threshold-based threat classification

## 📁 Project Structure

```
ids-system/
├── advanced_ids_app.py          # Main Streamlit application
├── setup_ids.py                 # Setup and verification script
├── requirements.txt             # Python dependencies
├── IDS_UI_GUIDE.md             # Detailed UI documentation
├── README.md                    # This file
├── models/
│   ├── intrusion-detection-system.ipynb  # Training notebook
│   └── ...                      # Other model files
└── data/
    └── checkpoints/
        ├── random_forest_binary_classifier.joblib
        ├── xgboost_binary_classifier.joblib
        ├── lstm_multi_class_classifier.h5
        ├── attack_type_label_encoder.joblib
        ├── attack_type_scaler.joblib
        └── processed_cicids_data.parquet
```

## 📖 Usage Examples

### Basic Analysis
```python
# Upload CSV file with network traffic data
# System automatically:
# 1. Preprocesses data (cleaning, normalization)
# 2. Runs 3 ML models in parallel
# 3. Generates ensemble predictions
# 4. Displays results with threat levels
```

### Batch Processing
```python
# Upload large CSV file (1000+ records)
# Set threat threshold (e.g., 50%)
# System processes entire batch
# Exports high-threat records separately
```

### Export Results
```python
# Download results as CSV for Excel/SQL analysis
# Download results as JSON for API integration
# Download high-threats report for incident response
```

## 🔧 Configuration

### Model Paths
Update paths in `advanced_ids_app.py`:
```python
output_dir = '/content/drive/My Drive/IDS_Checkpoints/'
```

### Threat Threshold
Adjust in the UI or modify default:
```python
confidence_threshold = st.slider("Threshold", 0, 100, 50)
```

### Port Configuration
```bash
streamlit run advanced_ids_app.py --server.port 8502
```

## 📊 Input Data Requirements

### CSV Format
- One network traffic record per row
- Approximately 80 numeric features
- Column headers (will be auto-cleaned)

### Recommended Features
- Flow duration and packet counts
- Protocol statistics (TCP/UDP)
- Payload bytes and flags
- Source/destination metrics
- Temporal characteristics
- Bidirectional flow information

## 📤 Output Examples

### Full Results
```
Record_ID | RF_Prediction | RF_Confidence | XGB_Prediction | Ensemble_Score | Attack_Type | Threat_Level
0         | Benign        | 95.2%         | Benign         | 12.3%          | N/A         | 🟢 LOW
1         | Attack        | 87.5%         | Attack         | 91.2%          | DoS         | 🔴 CRITICAL
```

### Threat Records (Filtered)
```
Record_ID | Threat_Score | Attack_Type | Confidence
1         | 91.2%        | DoS         | 94.5%
5         | 78.5%        | PortScan    | 87.2%
12        | 65.3%        | Intrusion   | 72.1%
```

## 🎓 Model Training

The models were trained on the **CICIDS2017** dataset:
- **Training data**: Network traffic with labeled attacks
- **Features**: 80 network traffic metrics
- **Attack types**: DoS, DDoS, PortScan, Brute Force, Infiltration, etc.
- **Class distribution**: Balanced benign and attack samples

## 🔒 Security & Privacy

- **Local Processing**: All data processed locally, no cloud uploads
- **No Persistence**: Results not permanently stored
- **Model Encryption**: Trained models protected
- **Privacy**: Only statistical summaries retained
- **Threshold Customization**: Adjust sensitivity for your environment

## 🚨 Alert System

### Critical Threats (> 80%)
- Immediate blocking recommended
- Manual investigation required
- Incident response activation

### High Threats (60-80%)
- Enhanced monitoring enabled
- Alert notifications sent
- Log for audit trail

### Medium Threats (40-60%)
- Continued observation
- Behavior analysis
- Pattern tracking

### Low Threats (< 40%)
- Regular monitoring
- Baseline comparison
- No immediate action

## 📊 Performance Metrics

The system tracks:
- **Detection Accuracy**: Percentage of correct classifications
- **Model Agreement**: Consensus between RF and XGBoost
- **Confidence Levels**: Average prediction certainty
- **False Positive Rate**: Incorrect attack classifications
- **Processing Speed**: Records per second

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not found | Ensure models are in `/IDS_Checkpoints/` directory |
| Slow processing | Use batch processing, reduce batch size |
| Poor predictions | Check input data quality, verify feature alignment |
| High false positives | Increase threat threshold |
| Installation errors | Install dependencies: `pip install -r requirements.txt` |

## 📚 Documentation

- **IDS_UI_GUIDE.md**: Detailed UI module documentation
- **Training Notebook**: `intrusion-detection-system.ipynb` for model details
- **Requirements.txt**: All Python dependencies

## 🛠️ System Requirements

- **OS**: Windows, macOS, Linux
- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for models and dependencies
- **GPU**: Optional (for faster inference)

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:
- Streamlit (UI framework)
- Pandas & NumPy (Data processing)
- scikit-learn (Random Forest)
- XGBoost (Gradient boosting)
- TensorFlow (Deep learning)
- Plotly (Interactive visualizations)
- Matplotlib & Seaborn (Static plots)

## 🎯 Use Cases

1. **Network Security**: Real-time intrusion detection
2. **Threat Hunting**: Identify suspicious patterns
3. **Incident Response**: Quick threat classification
4. **Security Audit**: Bulk traffic analysis
5. **Research**: Machine learning on network data
6. **Training**: Educational IDS demonstrations

## 📈 Future Enhancements

- Real-time streaming data support
- Database integration for persistent logging
- Advanced alerting system
- Custom model training interface
- API endpoint for programmatic access
- Docker containerization
- Distributed processing support

## 📞 Support

For issues or questions:
1. Check the **IDS_UI_GUIDE.md** documentation
2. Review **troubleshooting** section above
3. Verify model file locations
4. Check system requirements
5. Review data format compliance

## 📜 Version History

- **v1.0** (Current): Advanced Professional Edition
  - Multi-module Streamlit UI
  - Ensemble ML models
  - Professional visualizations
  - Batch processing
  - History tracking

## 📄 License & Attribution

This project uses:
- CICIDS2017 Dataset: Canadian Institute for Cybersecurity
- Streamlit: Open-source app framework
- scikit-learn: ML library
- TensorFlow/Keras: Deep learning framework

## 🙏 Acknowledgments

Built with state-of-the-art ML and cybersecurity best practices.

---

## 🛡️ Quick Reference

**Launch App**:
```bash
streamlit run advanced_ids_app.py
```

**Verify Setup**:
```bash
python setup_ids.py --verify-only
```

**Install Dependencies**:
```bash
pip install -r requirements.txt
```

**Access App**:
```
http://localhost:8501
```

---

**Your network is now protected with enterprise-grade AI threat detection! 🛡️**

*Advanced Intrusion Detection System | Version 1.0*
