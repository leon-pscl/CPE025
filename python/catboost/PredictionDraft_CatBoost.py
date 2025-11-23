# catboost_predict_water_class_with_input.py

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib

model_path = r"catboost_water_model.cbm"
scaler_path = r"catboost_colorimetry_scaler.pkl"
feature_names_path = r"catboost_feature_names.pkl"
cluster_map_path = r"cluster_names/cluster_to_class_name.pkl"

def load_artifacts():
    model = CatBoostClassifier()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    cluster_map = joblib.load(cluster_map_path)
    return model, scaler, feature_names, cluster_map

def predict_from_values(values_dict):
    model, scaler, feature_names, cluster_map = load_artifacts()
    row = {f: values_dict.get(f, 0.0) for f in feature_names}
    X_input = pd.DataFrame([row])
    X_scaled = scaler.transform(X_input)
    pred_cluster = int(model.predict(X_scaled)[0].item())

    # Convert predicted cluster to class name
    class_name = cluster_map.get(pred_cluster, f"Unknown class ({pred_cluster})")
    return class_name

if __name__ == "__main__":
    model, scaler, feature_names, cluster_map = load_artifacts()
    print("Please input the following water colorimetry concentration values:")
    user_inputs = {}
    for feature in feature_names:
        while True:
            try:
                val = float(input(f"{feature}: "))
                user_inputs[feature] = val
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    predicted_class_name = predict_from_values(user_inputs)
    print("Predicted water class:", predicted_class_name)
