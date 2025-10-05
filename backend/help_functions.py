from http.client import HTTPException
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


def get_avaliable_cols_from(dataFrame: pandas.DataFrame, mode: Union["api", "console"] = "console") -> list[str]:
    cols_to_save = ['kepid', 'kepler_name', 'predicted_class', 'prediction_probability']
    available_cols = [col for col in cols_to_save if col in dataFrame.columns]

    if mode == "api":
        if not available_cols:
            raise HTTPException(status_code=500, detail="In the results, no desired columns were found")
        for col in cols_to_save:
            if col not in available_cols:
                raise Warning(f"Info: Column '{col}' was not found in the results")
    elif mode == "console":     
        if not available_cols:
            raise ValueError("Error: None of the desired columns were found in the results")
        for col in cols_to_save:
            if col not in available_cols:
                raise Warning(f"Info: Column '{col}' was not found in the results")
    return available_cols