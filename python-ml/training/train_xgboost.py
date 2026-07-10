"""
XGBoost Fraud Model Training
==============================
Generates synthetic training data based on the patterns defined in
TestDataGenerator.java and trains an XGBoost fraud classifier.

Run this once before starting the inference consumer:
    python training/train_xgboost.py

Output:
    models/fraud_xgb.pkl — trained XGBoost model
    models/scaler.pkl — feature scaler
    mlruns/ — MLflow experiment tracking
"""

import os

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import structlog
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

# Feature columns - must match FEATURE_COLUMNS in features/engineer.py exactly
FEATURE_COLUMNS = [
    "amount_ratio", "daily_limit_ratio", "is_amount_unusual", "velocity_count", "velocity_squared", "is_high_velocity",
    "is_unkown_location", "is_primary_location", "is_different_city", "is_online", "is_typical_category",
    "is_suspicious_merchant",
    "hour", "is_off_hours", "is_weekend", "is_high_risk_customer", "is_low_risk_customer", "is_bot_device",
    "is_rapid_fire"
]

def generate_synthetic_data(n_samples: int = 10000) -> tuple:
    """
    Generate synthetic training data based on the patterns defined in TestDataGenerator.java

     Three transaction types match Java scenarios:
    - Normal (80%) → legitimate, low risk
    - High velocity (10%) → card testing attack, high risk
    - Unusual amount (10%) → fraud, high risk

    :param n_samples: number of samples to generate
    :return: X and y arrays
    """
    np.random.seed(42)
    X, y = [], []

    n_normal = int(n_samples * 0.8)
    n_velocity = int(n_samples * 0.1)
    n_unusual = int(n_samples * 0.1)

    # -------Normal transactions (legitimate)-------
    for _ in range(n_normal):
        features = {
            "amount_ratio": np.random.uniform(0.3, 1.8),
            "daily_limit_ratio": np.random.uniform(0.01, 0.3),
            "is_amount_unusual": np.random.choice([0, 1], p=[0.92, 0.08]),
            "velocity_count": np.random.randint(1, 3),
            "velocity_squared": np.random.uniform(1, 9),
            "is_high_velocity": 0,
            "is_unkown_location": np.random.choice([0, 1], p=[0.9, 0.1]),
            "is_primary_location": np.random.choice([0, 1], p=[0.2, 0.8]),  # 20% chance isn't primary
            "is_different_city": np.random.choice([0, 1], p=[0.9, 0.1]),
            "is_online": np.random.randint(0, 2),
            "is_typical_category": np.random.choice([0, 1], p=[0.15, 0.85]),  # 15% chance unusual,
            "is_suspicious_merchant": np.random.choice([0, 1], p=[0.95, 0.05]),
            "hour": np.random.randint(8, 22),
            "is_off_hours": 0,
            "is_weekend": np.random.randint(0, 2),
            "is_high_risk_customer": np.random.choice([0, 1], p=[0.85, 0.15]),
            "is_low_risk_customer": np.random.choice([0, 1], p=[0.15, 0.85]),
            "is_bot_device": np.random.choice([0, 1], p=[0.97, 0.03]),
            "is_rapid_fire": np.random.choice([0, 1], p=[0.97, 0.03]),
        }
        X.append([features[col] for col in FEATURE_COLUMNS])
        y.append(0)  # legitimate

    # -------High-velocity attack (card testing)-------
    for _ in range(n_velocity):
        velocity = np.random.randint(5, 15)
        features = {
            "amount_ratio": np.random.uniform(0.05, 0.3),  # smaller amounts
            "daily_limit_ratio": np.random.uniform(0.01, 0.1),
            "is_amount_unusual": np.random.choice([0, 1], p=[0.4, 0.6]),
            "velocity_count": velocity,
            "velocity_squared": velocity ** 2,
            "is_high_velocity": 1,
            "is_unkown_location": np.random.choice([0, 1], p=[0.3, 0.7]),
            "is_primary_location": np.random.choice([0, 1], p=[0.75, 0.25]),  # 25% chance primary
            "is_different_city": np.random.choice([0, 1], p=[0.4, 0.6]),
            "is_online": 1,
            "is_typical_category": np.random.choice([0, 1], p=[0.75, 0.25]),  # 25% chance typical
            "is_suspicious_merchant": np.random.choice([0, 1], p=[0.2, 0.8]),
            "hour": np.random.randint(0, 24),
            "is_off_hours": np.random.randint(0, 2),
            "is_weekend": np.random.randint(0, 2),
            "is_high_risk_customer": np.random.choice([0, 1], p=[0.3, 0.7]),
            "is_low_risk_customer": np.random.choice([0, 1], p=[0.8, 0.2]),
            "is_bot_device": np.random.choice([0, 1], p=[0.2, 0.8]),
            "is_rapid_fire": np.random.choice([0, 1], p=[0.15, 0.85]),
        }
        X.append([features[col] for col in FEATURE_COLUMNS])
        y.append(1)  # fraud - card testing

    # ---------Unusual amount fraud -----------
    for _ in range(n_unusual):
        features = {
            "amount_ratio": np.random.uniform(4.0, 10.0),
            "daily_limit_ratio": np.random.uniform(0.5, 1.5),
            "is_amount_unusual": 1,
            "velocity_count": np.random.randint(1, 4),
            "velocity_squared": np.random.uniform(1, 16),
            "is_high_velocity": 0,
            "is_unkown_location": np.random.randint(0, 2),
            "is_primary_location": 0,
            "is_different_city": np.random.choice([0, 1], p=[0.35, 0.65]),
            "is_online": 1,
            "is_typical_category": np.random.choice([0, 1], p=[0.7, 0.3]),
            "is_suspicious_merchant": np.random.randint(0, 2),
            "hour": np.random.randint(0, 24),
            "is_off_hours": np.random.randint(0, 2),
            "is_weekend": np.random.randint(0, 2),
            "is_high_risk_customer": np.random.choice([0, 1], p=[0.3, 0.7]),
            "is_low_risk_customer": np.random.choice([0, 1], p=[0.8, 0.2]),
            "is_bot_device": np.random.choice([0, 1], p=[0.6, 0.4]),
            "is_rapid_fire": np.random.choice([0, 1], p=[0.7, 0.3]),
        }
        X.append([features[col] for col in FEATURE_COLUMNS])
        y.append(1)  # fraud

    return np.array(X), np.array(y)


def train():
    os.makedirs("models", exist_ok=True)
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("fraud-xgboost")

    logger.info("Generating synthetic training data")
    x, y = generate_synthetic_data(n_samples=10000)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logger.info("Training XGBoost model", training_samples=len(x_train), test_samples=len(x_test),
                fraud_rate=f"{y.mean():.1%}")

    with mlflow.start_run(run_name="xgb-v1"):
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 4,  # handles class imbalance (80/20 split)
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        mlflow.log_params(params)

        # Train
        model = xgb.XGBClassifier(**params, eval_metric="auc")
        model.fit(x_train_scaled, y_train, eval_set=[(x_test_scaled, y_test)], verbose=False)

        # Evaluate
        y_pred = model.predict(x_test_scaled)
        y_prob = model.predict_proba(x_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metrics({
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        # Log feature importance
        importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

        print("\n----Model Performance-----------------------")
        print(f"AUC: {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\n----Top 5 Important Features---------------")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")
        print("\n---Classification Report----------\n")
        print(classification_report(y_test, y_pred, target_names=["legitimate", "fraud"]))

        # Save model and scaler
        joblib.dump(model, "models/fraud_xgb.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        mlflow.xgboost.log_model(model, "model")

        logger.info("model_saved", model_path="models/fraud_xgb.pkl", auc=round(auc, 4))

    print("\nModel saved to models/fraud_xgb.pkl")
    print("Scaler saved to models/scaler.pkl")
    print("MLflow run logger to mlruns/xgb-v1")
    print("\nNext: python consumers/inference_consumer.py")

if __name__ == "__main__":
    train()
