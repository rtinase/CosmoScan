import pickle
from typing import Any

def save_model(model, scaler, features):
    print("PROCESS START: model, scaler, and features are being saved...")

    with open('exoplanet_model.pkl', 'wb') as f:
        pickle.dump(model, f)   # move to joblib
    
    with open('exoplanet_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('exoplanet_features.pkl', 'wb') as f:
        pickle.dump(features, f)

    print("PROCESS END: model, scaler, and features have been saved.")

def load_model() -> tuple[Any, Any, Any]:
    print("PROCESS START: model, scaler, and features are being loaded...")

    with open('exoplanet_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('exoplanet_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('exoplanet_features.pkl', 'rb') as f:
        features = pickle.load(f)

    print("PROCESS END: model, scaler, and features have been loaded.")

    return model, scaler, features