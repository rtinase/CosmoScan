# ml.py
import pandas
from help_functions import load_model, load_data

def predict(file_path) -> pandas.DataFrame:
    model, scaler, features = load_model()
    new_data = load_data(file_path, "semicolon")

    for col in features:
        if col in new_data.columns:
            new_data[col] = new_data[col].fillna(new_data[col].mean())
        else:
            return None

    X_new = new_data[features]
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)[:, 1]

    new_data['predicted_class'] = predictions
    new_data['prediction_probability'] = probabilities

    if 'koi_disposition' in new_data.columns:
        new_data['actual_class'] = new_data['koi_disposition'].apply(
            lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
        )

    return new_data
