
# 🚀 Optimized Steganalysis Deployment Pipeline
import numpy as np
import cv2
import joblib
import pickle
from pathlib import Path
from typing import Optional, Tuple

class OptimizedSteganalysisDetector:
    """Production-ready steganalysis detector with maximum accuracy"""
    
    def __init__(self, models_dir: str):
        """Initialize detector with saved models"""
        self.models_dir = Path(models_dir)
        self._load_models()
    
    def _load_models(self):
        """Load all saved model components"""
        try:
            # Load main model
            self.model = joblib.load(self.models_dir / 'model_akhir.pkl')
            print(f"✅ Main model loaded: {type(self.model).__name__}")
            
            # Optional: Load preprocessing components (if you saved them)
            try:
                self.scaler = joblib.load(self.models_dir / 'feature_scaler_akhir.pkl')
                print("✅ Feature scaler loaded")
            except:
                self.scaler = None
                print("⚠️ Feature scaler not found")
            
            try:
                self.selector = joblib.load(self.models_dir / 'feature_selector_akhir.pkl')
                print("✅ Feature selector loaded")
            except:
                self.selector = None
                print("⚠️ Feature selector not found")
            
            print("✅ All available models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def predict_features(self, features_588: np.ndarray) -> Tuple[int, float]:
        """Predict using a provided 588-dim raw SRM feature vector"""
        if features_588.ndim == 1:
            features_588 = features_588.reshape(1, -1)
        # Directly use the trained model (it already encapsulates preprocessing inside training)
        pred = self.model.predict(features_588)[0]
        if hasattr(self.model, 'predict_proba'):
            conf = float(self.model.predict_proba(features_588)[0].max())
        else:
            conf = 0.5
        return int(pred), conf
