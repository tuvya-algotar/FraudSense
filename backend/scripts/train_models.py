"""
train_models.py
FraudSense — ML Model Training Pipeline

Generates realistic synthetic transactions safely aligned with z-score normalization
and trains a RandomForest classifier matching strictly 8 features.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data.csv"
MODEL_PATH = "models/fraud_model.pkl"

FEATURE_COLS = [
    "amount", "amount_deviation", "location_change", "new_device",
    "merchant_risk", "txn_velocity", "is_night", "device_change_frequency"
]

def generate_synthetic_data(num_rows=2500):
    print(f"Generating {num_rows} synthetic transactions...")
    np.random.seed(42)

    data = []
    for _ in range(num_rows):
        is_fraud_scenario = np.random.rand() < 0.30

        if not is_fraud_scenario:
            # ── LEGITIMATE TRANSACTION ────────────────────────────────────
            amount = np.random.uniform(10, 15000)
            
            # z-score normalization bounded
            mean = np.random.uniform(10, 15000)
            std = mean * np.random.uniform(0.1, 0.4)
            z_score = (amount - mean) / max(std, 1.0)
            amount_deviation = max(0.0, min(z_score, 10.0))
            
            location_change = np.random.choice([0.0, 0.5], p=[0.9, 0.1])
            new_device = np.random.choice([0.0, 0.5], p=[0.9, 0.1])
            merchant_risk = 0 if np.random.rand() < 0.95 else 1
            txn_velocity = np.random.randint(1, 4)
            is_night = 0 if np.random.rand() < 0.88 else 1
            device_change_frequency = 0 if np.random.rand() < 0.85 else np.random.randint(1, 3)
        else:
            # ── FRAUDULENT TRANSACTION ────────────────────────────────────
            fraud_type = np.random.choice(["high_amount", "location", "device", "velocity", "mixed"])

            if fraud_type == "high_amount":
                amount = np.random.uniform(50001, 100000)
                location_change = np.random.choice([0.0, 0.5])
                new_device = np.random.choice([0.0, 0.5, 1.0])
                merchant_risk = int(np.random.rand() < 0.5)
                txn_velocity = np.random.randint(1, 8)
            elif fraud_type == "location":
                amount = np.random.uniform(5000, 60000)
                location_change = 1.0
                new_device = np.random.choice([0.5, 1.0])
                merchant_risk = int(np.random.rand() < 0.4)
                txn_velocity = np.random.randint(2, 7)
            elif fraud_type == "device":
                amount = np.random.uniform(2000, 80000)
                location_change = np.random.choice([0.0, 0.5, 1.0])
                new_device = 1.0
                merchant_risk = int(np.random.rand() < 0.5)
                txn_velocity = np.random.randint(1, 6)
            elif fraud_type == "velocity":
                amount = np.random.uniform(1000, 50000)
                location_change = np.random.choice([0.0, 0.5])
                new_device = np.random.choice([0.0, 0.5])
                merchant_risk = int(np.random.rand() < 0.4)
                txn_velocity = np.random.randint(7, 10)
            else:  # mixed
                amount = np.random.uniform(30000, 100000)
                location_change = 1.0
                new_device = 1.0
                merchant_risk = 1
                txn_velocity = np.random.randint(5, 10)

            # Simulated aggressive deviations
            mean = np.random.uniform(1000, 10000)
            std = mean * np.random.uniform(0.1, 0.3)
            z_score = (amount - mean) / max(std, 1.0)
            amount_deviation = max(0.0, min(z_score, 10.0))

            is_night = int(np.random.rand() < 0.55)
            device_change_frequency = np.random.randint(1, 4)

        # ── DETERMINISTIC LABEL ───────────────────────────────────────────
        label = 1 if (
            amount_deviation > 4.0 or
            location_change == 1.0 or
            new_device == 1.0 or
            txn_velocity > 6 or
            merchant_risk == 1
        ) else 0

        data.append([
            round(amount, 2),
            round(amount_deviation, 4),
            location_change,
            new_device,
            merchant_risk,
            txn_velocity,
            is_night,
            device_change_frequency,
            label
        ])

    df = pd.DataFrame(data, columns=FEATURE_COLS + ["label"])
    df.to_csv(DATA_PATH, index=False)

    fraud_count = df["label"].sum()
    print(f"Dataset saved to {DATA_PATH}")
    print(f"  Total: {len(df)} | Fraud: {fraud_count} ({fraud_count/len(df)*100:.1f}%) | Legit: {len(df)-fraud_count}")

    return df

def main():
    # ── 1. Generate new structured dataset ────────────────────────────────────
    # Forces generation to ensure strict adherence to new feature sizes and logic
    df = generate_synthetic_data(2500)

    # ── 2. Feature / label split ──────────────────────────────────────────────
    X = df[FEATURE_COLS]
    y = df["label"]

    assert X.shape[1] == 8, f"Expected 8 features, got {X.shape[1]}"
    print(f"Feature matrix: {X.shape}, Label distribution: {dict(y.value_counts())}")

    # ── 3. Train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 4. Train RandomForestClassifier (with class weights) ──────────────────
    print("Training RandomForestClassifier (200 trees, class_weight='balanced')...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── 5. Evaluate and Print validation info ─────────────────────────────────
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Model Accuracy (Basic): {acc * 100:.2f}%")
    print(f"{'='*50}")
    
    print("\nFEATURE RANGES:")
    for col in FEATURE_COLS:
        print(f"  - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    print(f"\n{'='*50}")
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # Verify feature count
    assert model.n_features_in_ == 8, f"Model trained on {model.n_features_in_} features, expected 8"

    # ── 6. Save model ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Features mapped strictly: {FEATURE_COLS}")

if __name__ == "__main__":
    main()