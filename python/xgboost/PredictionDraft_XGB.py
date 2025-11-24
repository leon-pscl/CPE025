# catboost_predict_water_class_with_input.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime

model_path = r"xgbb_water_model.pkl"
scaler_path = r"xgb_colorimetry_scaler.pkl"
feature_names_path = r"xgb_feature_names.pkl"
cluster_map_path = r"cluster_names/cluster_to_class_name.pkl"

def load_artifacts():
    model = joblib.load(model_path)  # Use joblib to load the whole sklearn model
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    cluster_map = joblib.load(cluster_map_path)
    return model, scaler, feature_names, cluster_map


def predict_from_values(values_dict):
    model, scaler, feature_names, cluster_map = load_artifacts()
    row = {f: values_dict.get(f, 0.0) for f in feature_names}
    X_input = pd.DataFrame([row], columns=feature_names)
    X_scaled = scaler.transform(X_input)
    pred_cluster = int(model.predict(X_scaled)[0].item())

    # Convert predicted cluster to class name
    class_name = cluster_map.get(pred_cluster, f"Unknown class ({pred_cluster})")
    return class_name


def save_result_to_csv(user_inputs, predicted_class_name, csv_path="predictions_log.csv"):

    # Combine input features and prediction into one dictionary
    row = dict(user_inputs)
    # Add ISO8601 timestamp
    row["Timestamp"] = datetime.now().isoformat(timespec='seconds')
    row["Predicted_Class"] = predicted_class_name

    # If the CSV does not exist, create it with headers
    if not os.path.exists(csv_path):
        df = pd.DataFrame([row])
        df.to_csv(csv_path, index=False)
    else:
        # If the CSV exists, append without headers
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode='a', header=False, index=False)



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

    # Save to CSV and print file size
    csv_path = "predictions_log.csv"
    save_result_to_csv(user_inputs, predicted_class_name, csv_path)
    file_size = os.path.getsize(csv_path)
    print(f"Prediction saved to {csv_path}")
    print(f"Current file size: {file_size} bytes")

