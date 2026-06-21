import subprocess
import sys
import time
import os

# Get the path to the streamlit app file
streamlit_app_path = "streamlit_app.py"

# Streamlit app code as a multiline string
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from io import StringIO

# --- Paths to saved models and preprocessors ---
output_dir = 'models'

rf_model_path = os.path.join(output_dir, 'random_forest_binary_classifier.joblib')
xgb_model_path = os.path.join(output_dir, 'xgboost_binary_classifier.joblib')
lstm_model_path = os.path.join(output_dir, 'lstm_multi_class_classifier.h5')
label_encoder_path = os.path.join(output_dir, 'attack_type_label_encoder.joblib')
scaler_path = os.path.join(output_dir, 'attack_type_scaler.joblib')
processed_df_path = os.path.join(output_dir, 'processed_cicids_data.parquet') # To get column names

# --- Load Models and Preprocessors ---
@st.cache_resource
def load_resources():
    try:
        rf_classifier = joblib.load(rf_model_path)
        xgb_classifier = joblib.load(xgb_model_path)
        lstm_model = tf.keras.models.load_model(lstm_model_path)
        label_encoder = joblib.load(label_encoder_path)
        scaler = joblib.load(scaler_path)

        # Load the full processed dataframe to correctly get the feature column names
        df_full = pd.read_parquet(processed_df_path)
        expected_columns = df_full.drop(['Label', 'Attack_Type_Original'], axis=1, errors='ignore').columns.tolist()

        return rf_classifier, xgb_classifier, lstm_model, label_encoder, scaler, expected_columns
    except Exception as e:
        st.error(f"Error loading resources: {e}. Please ensure all models and preprocessors are saved in {output_dir}")
        st.stop()

rf_classifier, xgb_classifier, lstm_model, label_encoder, scaler, expected_columns = load_resources()

# --- Preprocessing Function (mimicking process_cicids_dataset) ---
def preprocess_input_data(df_raw):
    df = df_raw.copy()

    # 1. Clean column names (mimic original processing)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.replace('.', '', regex=False)

    # 2. Convert object columns to numeric where possible
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Replace infinite values with NaN
    df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)

    # 4. Handle missing values (simple fillna for UI context, ideally use training medians/modes)
    # For this UI demo, we'll fill with 0s for simplicity if NaNs persist. 
    # In a production system, you'd use the medians/modes from the training data.
    df = df.fillna(0)

    # 5. Ensure consistent columns for binary classification (X)
    # Drop 'Attack_Type_Original' if present in raw input, as it's not a feature for binary. 'Label' also.
    X_binary_cols = [col for col in df.columns if col not in ['Label', 'Attack_Type_Original']]
    X_binary = df[X_binary_cols]
    
    # Align columns with those used during training (expected_columns comes from the original processed data)
    missing_cols_binary = set(expected_columns) - set(X_binary.columns)
    for c in missing_cols_binary:
        X_binary[c] = 0 # Add missing columns, fill with 0
    X_binary = X_binary[expected_columns] # Reorder to match training data

    # 6. Prepare data for multi-class LSTM (X_attack_scaled_reshaped)
    # The scaler expects the same features as X_attack during training.
    X_multiclass = X_binary.copy() # Use the cleaned binary features as base
    X_multiclass_scaled = scaler.transform(X_multiclass)
    num_features = X_multiclass_scaled.shape[1]
    X_multiclass_reshaped = X_multiclass_scaled.reshape(X_multiclass_scaled.shape[0], 1, num_features)

    return X_binary, X_multiclass_reshaped


# --- Streamlit UI --- 
st.set_page_config(layout="wide")
st.title("Intrusion Detection System (IDS) Demo")
st.markdown("Upload a CSV file containing network traffic data to get intrusion predictions.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df_raw = pd.read_csv(data)

    st.subheader("Uploaded Data (First 5 Rows)")
    st.dataframe(df_raw.head())

    st.subheader("Prediction Results")

    if st.button("Run Predictions"):
        with st.spinner("Preprocessing data and making predictions..."):
            X_binary, X_multiclass_reshaped = preprocess_input_data(df_raw)

            # Binary Classification Predictions
            st.write("#### Binary Classification (Benign vs. Attack)")
            
            # Random Forest
            binary_pred_rf = rf_classifier.predict(X_binary)
            binary_proba_rf = rf_classifier.predict_proba(X_binary)[:, 1]
            
            # XGBoost
            binary_pred_xgb = xgb_classifier.predict(X_binary)
            binary_proba_xgb = xgb_classifier.predict_proba(X_binary)[:, 1]

            # Multi-class Classification Predictions (LSTM)
            st.write("#### Multi-class Classification (Specific Attack Type)")
            multi_pred_proba_lstm = lstm_model.predict(X_multiclass_reshaped, verbose=0)
            multi_pred_lstm_encoded = np.argmax(multi_pred_proba_lstm, axis=1)
            multi_pred_lstm_labels = label_encoder.inverse_transform(multi_pred_lstm_encoded)

            # Compile results into a DataFrame
            results_df = pd.DataFrame({
                'Record_ID': df_raw.index,
                'RF_Binary_Prediction': np.where(binary_pred_rf == 1, 'Attack', 'Benign'),
                'RF_Attack_Probability': binary_proba_rf.round(4),
                'XGB_Binary_Prediction': np.where(binary_pred_xgb == 1, 'Attack', 'Benign'),
                'XGB_Attack_Probability': binary_proba_xgb.round(4),
                'LSTM_Attack_Type': multi_pred_lstm_labels,
                'LSTM_Confidence': np.max(multi_pred_proba_lstm, axis=1).round(4)
            })
            
            # Optional: Add original 'Label' and 'Attack_Type_Original' if they exist in raw data
            if 'Label' in df_raw.columns:
                results_df['Original_Label'] = df_raw['Label']
            if 'Attack_Type_Original' in df_raw.columns:
                results_df['Original_Attack_Type'] = df_raw['Attack_Type_Original']

            st.dataframe(results_df)

            # --- Visualization (Summary) ---
            st.write("#### Prediction Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Random Forest Binary Predictions:**")
                st.write(results_df['RF_Binary_Prediction'].value_counts())
                st.write("**XGBoost Binary Predictions:**")
                st.write(results_df['XGB_Binary_Prediction'].value_counts())
            with col2:
                st.write("**LSTM Multi-class Predictions:**")
                st.write(results_df['LSTM_Attack_Type'].value_counts())

            # Plotting (e.g., distribution of attack types for LSTM)
            st.write("##### LSTM Predicted Attack Type Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            results_df['LSTM_Attack_Type'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Predicted Attack Types (LSTM)')
            ax.set_xlabel('Attack Type')
            ax.set_ylabel('Count')
            st.pyplot(fig)

else:
    st.info("Please upload a CSV file to get started. You can use a sample from your training data, or a new dataset with similar features.")


# Write the Streamlit app code to a file
with open(streamlit_app_path, "w") as f:
    f.write(streamlit_app_code)

# Kill any previously running Streamlit processes to free up the port
print("Attempting to kill any existing Streamlit processes...")
subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
time.sleep(1) # Give it a moment to terminate

# Run Streamlit in the background and print its public URL
print("Launching Streamlit app...")
# Use 'nohup' and '&' to run in the background and prevent output to stdout
# We capture output to file to avoid clogging Colab's output, but ensure URL is printed.
process = subprocess.Popen(["nohup", "streamlit", "run", streamlit_app_path, "--browser.gatherUsageStats", "false", "--server.port", "8501", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Give Streamlit a moment to start and generate the URL
time.sleep(5) # Initial wait

# Attempt to find the public URL in the logs
url_found = False
for _ in range(15): # Check multiple times with slightly longer total timeout
    stdout_line = process.stdout.readline()
    if "External URL" in stdout_line:
        print(stdout_line.strip())
        url_found = True
        break
    elif process.poll() is not None: # If process exited early
        print("Streamlit process exited unexpectedly.")
        stderr_output = process.stderr.read()
        print("Stderr:", stderr_output)
        break
    time.sleep(1)

if not url_found:
    print("Could not find Streamlit public URL after multiple attempts. It might be starting slowly or encountered an error.")
    print("You can try to find the URL manually in the nohup.out file or by checking `!pgrep -f streamlit` and `!cat nohup.out`")

print("\nStreamlit app is running in the background. If you don't see a public URL, try executing this cell again after some time.")
print("To stop the Streamlit app, you can use `!pkill -f streamlit`")
