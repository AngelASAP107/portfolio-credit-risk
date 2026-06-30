import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'lgbm_credit_risk.pkl'
METADATA_PATH = Path(__file__).parent.parent.parent / 'models' / 'model_metadata.json'


def load_model():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    model = joblib.load(MODEL_PATH)
    return model, metadata


def predict(model, metadata, input_data: dict) -> float:
    """Construye el DataFrame con todas las features en el orden correcto y predice."""
    feature_names = metadata['feature_names']
    cat_cols = metadata['preprocessing']['categorical_cols']

    # Crear fila con todas las features en None por defecto
    row = {feat: np.nan for feat in feature_names}

    # Llenar con los valores que llegaron en el request
    for key, value in input_data.items():
        if key in row:
            row[key] = value

    df = pd.DataFrame([row])

    # Convertir numéricas a float explícitamente — evita dtype object cuando el valor es None
    for col in df.columns:
        if col not in cat_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convertir categóricas al mismo dtype que el modelo espera
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    proba = model.predict_proba(df)[0, 1]
    return float(proba)
