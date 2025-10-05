from typing import Any, Union
import joblib
import os

import pandas

def save_model(model, scaler, features):
    print("PROCESS START: model, scaler, and features are being saved...")

    os.makedirs('./models', exist_ok=True)

    joblib.dump(model, './models/exoplanet_model.joblib')
    joblib.dump(scaler, './models/exoplanet_scaler.joblib')
    joblib.dump(features, './models/exoplanet_features.joblib')

    print("PROCESS END: model, scaler, and features have been saved.")

def load_model() -> tuple[Any, Any, Any]:
    print("PROCESS START: model, scaler, and features are being loaded...")

    model = joblib.load('./models/exoplanet_model.joblib')
    scaler = joblib.load('./models/exoplanet_scaler.joblib')
    features = joblib.load('./models/exoplanet_features.joblib')

    print("PROCESS END: model, scaler, and features have been loaded.")

    return model, scaler, features
    
def load_data(file_path: str, sign: Union["comma", "semicolon"]) -> pandas.DataFrame:
    print("PROCESS START: reading data...\n")
    if sign == "comma":
        dataFrame = pandas.read_csv(file_path, comment='#')
    elif sign == "semicolon":
        dataFrame = pandas.read_csv(file_path, sep=';', comment='#')
    else:
        raise ValueError("Unknown delimiter")
    print("PROCESS END: reading data...\n")
    return dataFrame    