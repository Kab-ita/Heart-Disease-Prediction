import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def _encode_dataframe(df, feature_columns, encoders=None, fit=False): 
    encoded = df[feature_columns].copy()
    if encoders is None:
        encoders = {}

    for col in feature_columns:
        if col in CATEGORICAL_FEATURES:
            if fit:
                le = LabelEncoder()
                encoded[col] = le.fit_transform(encoded[col].astype(str))
                encoders[col] = le
            else:
                le = encoders[col]
                seen = set(le.classes_) 
                encoded[col] = encoded[col].astype(str).apply(
                    lambda v: v if v in seen else le.classes_[0]
                )
                encoded[col] = le.transform(encoded[col])
        else:
            encoded[col] = encoded[col].astype(float)

    return encoded, encoders


def build_rf_model(train_df, feature_columns, target_column, n_estimators=200, random_state=42): 
    X_encoded, encoders = _encode_dataframe(train_df, feature_columns, fit=True)
    y = train_df[target_column].astype(int)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_encoded, y)
    return model, encoders


def predict_rf(model, encoders, feature_columns, sample_dict): 
    row_df = pd.DataFrame([sample_dict])[feature_columns]
    X_encoded, _ = _encode_dataframe(row_df, feature_columns, encoders=encoders, fit=False)

    pred = int(model.predict(X_encoded)[0])
    proba = model.predict_proba(X_encoded)[0]
    confidence = round(float(max(proba)) * 100, 2)

    importances = model.feature_importances_
    top_idx = importances.argsort()[::-1][:3]
    path = [
        f"{feature_columns[i]} = {sample_dict.get(feature_columns[i])} "
        f"(global feature importance: {round(importances[i] * 100, 1)}%)"
        for i in top_idx
    ]
    path.append(
        f"{model.n_estimators} trees voted -> {confidence}% confidence for class {pred} "
        f"(ensemble average, not a single traceable rule path)"
    )
    return pred, path, confidence