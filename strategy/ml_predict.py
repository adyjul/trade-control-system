import os
import pickle
import pandas as pd

# Global model cache
_model = None
_model_path = os.path.join(os.path.dirname(__file__), "breakout_rf_model.pkl")

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(_model_path):
            raise FileNotFoundError(f"Model file not found: {_model_path}")
        with open(_model_path, "rb") as f:
            _model = pickle.load(f)
    return _model

def is_signal_valid(features_dict: dict, use_ml=True) -> bool:
    
    if not use_ml:
        return True

    model = load_model()

    # Convert dict to DataFrame with one row
    df = pd.DataFrame([features_dict])

    try:
        prediction = model.predict(df)[0]
        return prediction == 1
    except Exception as e:
        print(f"[ML Filter] Prediction failed: {e}")
        return False
