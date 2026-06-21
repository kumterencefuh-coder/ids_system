#!/usr/bin/env python3
"""
Quick Start Setup for Advanced IDS System
Configures paths and launches the Streamlit application
"""

import os
import sys
import subprocess
import json
from pathlib import Path

class IDSSetup:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        self.config = {
            'app_name': 'Advanced Intrusion Detection System',
            'version': '1.0',
            'models_dir': str(base_dir / 'models'),
            'scalers_dir': str(base_dir / 'scalers'),
            'data_dir': str(base_dir / 'data' / 'checkpoints'),
            'port': 8501,
            'debug': False
        }
    
    def check_dependencies(self):
        """Verify all required packages are installed"""
        package_map = {
            'streamlit': 'streamlit',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'sklearn': 'scikit-learn',
            'xgboost': 'xgboost',
            'tensorflow': 'tensorflow',
            'plotly': 'plotly',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'joblib': 'joblib'
        }
        
        print("🔍 Checking dependencies...")
        missing = []
        
        for import_name, pip_name in package_map.items():
            try:
                __import__(import_name)
                print(f"✅ {pip_name}")
            except ImportError:
                print(f"❌ {pip_name}")
                missing.append(pip_name)
        
        if missing:
            print(f"\n⚠️ Missing packages: {', '.join(missing)}")
            print("Installing missing packages...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install packages: {e}")
                return False
            print("✅ Installation complete. Verifying imports...")
            missing = []
            for import_name, pip_name in package_map.items():
                try:
                    __import__(import_name)
                    print(f"✅ {pip_name}")
                except ImportError:
                    print(f"❌ {pip_name}")
                    missing.append(pip_name)
            if missing:
                print(f"\n❌ Still missing packages after install: {', '.join(missing)}")
                return False
            print("\n✅ All dependencies satisfied!")
            return True
        else:
            print("\n✅ All dependencies satisfied!")
            return True
    
    def verify_models(self):
        """Check if all model files exist"""
        print("\n🔍 Verifying model files...")
        
        required_paths = [
            (os.path.join(self.config['models_dir'], 'random_forest_binary_classifier.joblib'), 'Random Forest model'),
            (os.path.join(self.config['models_dir'], 'xgboost_binary_classifier.joblib'), 'XGBoost model'),
            (os.path.join(self.config['models_dir'], 'lstm_multi_class_classifier.h5'), 'LSTM model'),
            (os.path.join(self.config['scalers_dir'], 'attack_type_scaler.joblib'), 'Attack type scaler')
        ]
        optional_paths = [
            (os.path.join(self.config['scalers_dir'], 'attack_type_label_encoder.joblib'), 'Attack type label encoder'),
            (os.path.join(self.config['data_dir'], 'processed_cicids_data.parquet'), 'Processed CICIDS data')
        ]
        
        all_exist = True
        for path, description in required_paths:
            if os.path.exists(path):
                print(f"✅ {description}")
            else:
                print(f"❌ {description} - NOT FOUND")
                all_exist = False
        
        for path, description in optional_paths:
            if os.path.exists(path):
                print(f"✅ {description}")
            else:
                print(f"⚠️ {description} - NOT FOUND (optional)")
        
        if not all_exist:
            print(f"\n⚠️ Some required resources are missing.")
            print("Make sure the models and scaler are placed in the local project folders.")
            return False
        
        print("\n✅ All required resources are present! Optional resources may still be absent.")
        return True
    
    def launch_app(self, app_path='advanced_ids_app.py'):
        """Launch the Streamlit application"""
        print(f"\n🚀 Launching {self.config['app_name']}...")
        
        cmd = [
            'streamlit',
            'run',
            app_path,
            f'--server.port={self.config["port"]}',
            '--browser.gatherUsageStats=false',
            '--server.enableXsrfProtection=false'
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error launching app: {e}")
            return False
        except FileNotFoundError:
            print("❌ Streamlit not found. Please install it: pip install streamlit")
            return False
        
        return True
    
    def setup(self, verify_only=False):
        """Run full setup"""
        print(f"🛡️ {self.config['app_name']} - Setup Wizard\n")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            print("\n❌ Failed to install dependencies")
            return False
        
        # Verify models
        if not self.verify_models():
            if not verify_only:
                print("\n⚠️ Continuing anyway... (models might be downloaded at runtime)")
            else:
                return False
        
        if verify_only:
            print("\n✅ Setup verification complete!")
            return True
        
        print("\n" + "=" * 50)
        print("🎯 Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("=" * 50)
        
        # Launch app
        return self.launch_app()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced IDS System Setup and Launch'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify setup without launching'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port for Streamlit app (default: 8501)'
    )
    parser.add_argument(
        '--app-path',
        default=os.path.join('models', 'advanced_ids_app.py'),
        help='Path to Streamlit app file'
    )
    
    args = parser.parse_args()
    
    setup = IDSSetup()
    setup.config['port'] = args.port
    
    success = setup.setup(verify_only=args.verify_only)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
