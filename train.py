import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split  

from ml_core.discretize import discretize_row
from ml_core.id3 import build_id3_tree, predict_id3
from ml_core.c45 import build_c45_tree, predict_c45
from ml_core.random_forest import build_rf_model, predict_rf
from ml_core.validation import performance_report, majority_vote


def run_pipeline():
    print("Starting the ID3 / C4.5 / Random Forest training pipeline...")

    csv_path = os.path.join("data", "heart.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing core raw data file at: {csv_path}")
    df = pd.read_csv(csv_path)

    print("\n--- [Step 1: Exploratory Data Analysis (EDA) Summary] ---")
    print(f"Dataset shape (rows, columns): {df.shape}")
    print("\nMissing value counts:")
    print(df.isnull().sum())

    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    print(f"\nDropped {initial_count - len(df)} duplicate row(s).")

    continuous_list = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    categorical_list = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

    for col in continuous_list:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in categorical_list:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    print("Missing value imputation complete.")

    target_column = "HeartDisease"
    all_features = [col for col in df.columns if col != target_column]

    print("\n--- [Step 2: Feature/Target Split Verified] ---")
    print(f"Input features: {all_features}")
    print(f"Target output: {target_column}\n")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
 
    id3_train_matrix = []
    for _, row in train_df.iterrows():
        binned = discretize_row(row.to_dict())
        id3_train_matrix.append([binned[feat] for feat in all_features] + [int(row[target_column])])
    id3_tree = build_id3_tree(id3_train_matrix, list(range(len(all_features))), len(all_features), all_features)
 
    c45_train_matrix = [list(row[all_features].values) + [int(row[target_column])] for _, row in train_df.iterrows()]
    c45_tree = build_c45_tree(c45_train_matrix, list(range(len(all_features))), len(all_features), all_features,
                               continuous_list)
 
    rf_model, rf_encoders = build_rf_model(train_df, all_features, target_column)

    id3_preds, c45_preds, rf_preds = [], [], []
    for _, row in test_df.iterrows():
        row_dict = row.to_dict()
        id3_pred, _ = predict_id3(id3_tree, discretize_row(row_dict))
        c45_pred, _ = predict_c45(c45_tree, row_dict)
        rf_pred, _, _ = predict_rf(rf_model, rf_encoders, all_features, row_dict)
        id3_preds.append(id3_pred)
        c45_preds.append(c45_pred)
        rf_preds.append(rf_pred)

    majority_preds = [
        majority_vote(a, b, c)["verdict"] for a, b, c in zip(id3_preds, c45_preds, rf_preds)
    ]

    pairwise_agreements = sum(1 for a, b in zip(id3_preds, c45_preds) if a == b)
    agreement_rate = round(100 * pairwise_agreements / len(id3_preds), 2) if id3_preds else 0.0

    unanimous_count = sum(1 for a, b, c in zip(id3_preds, c45_preds, rf_preds) if a == b == c)
    unanimous_rate = round(100 * unanimous_count / len(id3_preds), 2) if id3_preds else 0.0

    id3_metrics = performance_report(test_df[target_column].tolist(), id3_preds)
    c45_metrics = performance_report(test_df[target_column].tolist(), c45_preds)
    rf_metrics = performance_report(test_df[target_column].tolist(), rf_preds)
    majority_metrics = performance_report(test_df[target_column].tolist(), majority_preds)

    print("\n================== PERFORMANCE METRICS ==================")
    print("ID3 metrics:           ", id3_metrics)
    print("C4.5 metrics:          ", c45_metrics)
    print("Random Forest metrics: ", rf_metrics)
    print("Majority-vote metrics: ", majority_metrics)
    print(f"\nID3 / C4.5 pairwise agreement rate on test set: {agreement_rate}% "
          f"({pairwise_agreements}/{len(id3_preds)} records)")
    print(f"ID3 / C4.5 / RF unanimous agreement rate: {unanimous_rate}% "
          f"({unanimous_count}/{len(id3_preds)} records)")
    print("===========================================================\n")

    os.makedirs("ml_models", exist_ok=True)
    with open("ml_models/id3_tree.json", "w") as f:
        json.dump(id3_tree, f)
    with open("ml_models/c45_tree.json", "w") as f:
        json.dump(c45_tree, f)
 
    joblib.dump(
        {"model": rf_model, "encoders": rf_encoders, "features": all_features},
        "ml_models/rf_model.joblib",
    )

    with open("ml_models/agreement_stats.json", "w") as f:
        json.dump({
            "agreement_rate": agreement_rate,    
            "unanimous_rate": unanimous_rate,    
            "test_set_size": len(id3_preds),
        }, f)
 
    with open("ml_models/metrics_report.json", "w") as f:
        json.dump({
            "id3": {"label": "Custom ID3 Algorithm (From Scratch)", **id3_metrics},
            "c45": {"label": "Custom C4.5 Engine Algorithm", **c45_metrics},
            "random_forest": {"label": "scikit-learn Random Forest (Reference Ensemble)", **rf_metrics},
            "majority_vote": {"label": "Best-of-3 Majority Vote (ID3 + C4.5 + RF)", **majority_metrics},
        }, f)

    print("Serialized id3_tree.json, c45_tree.json, rf_model.joblib, agreement_stats.json, "
          "and metrics_report.json to /ml_models/")


if __name__ == "__main__":
    run_pipeline()