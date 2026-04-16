"""
backend/services/ml_service.py
FraudSense — ML Prediction Service

Loads the RandomForest model trained by train_models.py and provides
predict_fraud() and explain_prediction() for the risk pipeline.

STRICT 8-feature contract:
  [amount, amount_deviation, location_change, new_device, merchant_risk,
   txn_velocity, is_night, device_change_frequency]
"""

import joblib
import os
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "models/fraud_model.pkl"

FEATURE_COLS = [
    "amount", "amount_deviation", "location_change", "new_device",
    "merchant_risk", "txn_velocity", "is_night", "device_change_frequency"
]


# ── Model singleton ───────────────────────────────────────────────────────────
model = None


def train_model():
    """Run the training pipeline to create a fresh model file."""
    print("⚙ Training model via backend/scripts/train_models.py ...")
    script_path = os.path.join("backend", "scripts", "train_models.py")
    subprocess.run([sys.executable, script_path], check=True)
    print("✅ Training complete.")


def _load_model():
    """Load the model from disk, training first if necessary."""
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"⚠  Model not found at {MODEL_PATH} — training now...")
        train_model()

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file {MODEL_PATH} not found even after training. "
            "Check train_models.py for errors."
        )

    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Validate feature count
    expected = len(FEATURE_COLS)
    actual = getattr(model, "n_features_in_", None)
    if actual is not None and actual != expected:
        raise RuntimeError(
            f"Model expects {actual} features but service provides {expected}. "
            "Delete models/fraud_model.pkl and restart to retrain."
        )

    logger.info(f"Loaded ML model from {MODEL_PATH} ({actual} features)")


# ── Eager load at import time ──────────────────────────────────────────────────
_load_model()


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(txn: dict, profile: dict) -> dict:
    """
    Extract the 8 features in STRICT order matching the training pipeline.
    """
    if not profile:
        profile = {}

    amount = float(txn.get("amount", 0))

    # 1. amount_deviation (z_score logic)
    mean = float(profile.get("avg_amount", 0.0))
    std = float(profile.get("std_amount", 0.0))
    if std > 0:
        z_score = (amount - mean) / std
        amount_deviation = max(0.0, min(z_score, 10.0))
    elif mean > 0:
        amount_deviation = min(amount / mean, 5.0)  # safe fallback cap
    else:
        amount_deviation = 0.0

    # 2. location_change
    location = txn.get("city", txn.get("location", "unknown")).lower().strip()
    frequent_locations = [loc.lower().strip() for loc in profile.get("frequent_locations", [])]
    
    if not frequent_locations:
        location_change = 0.5  # neutral score, no history
    elif not location or location == "unknown":
        location_change = 0.5  # moderate risk for unknown city
    elif location in frequent_locations:
        location_change = 0.0
    else:
        location_change = 1.0

    # 3. new_device
    device = txn.get("device_id", txn.get("device", "unknown"))
    trusted_devices = profile.get("trusted_devices", [])
    
    if not trusted_devices:
        new_device = 0.5 # fallback safely
    else:
        new_device = 0.0 if device in trusted_devices else 1.0

    # 4. merchant_risk
    merchant = str(txn.get("merchant_category", txn.get("merchant", "unknown"))).lower()
    RISKY_MERCHANTS = {"crypto", "unknown", "high-risk", "cryptocurrency", "wire_transfer", "gambling", "casino", "forex"}
    merchant_risk = 1 if merchant in RISKY_MERCHANTS else 0

    # 5. txn_velocity
    txn_velocity = int(txn.get("velocity_count_1h", 1))

    # 6. is_night
    hour = 12
    if "timestamp" in txn and txn["timestamp"]:
        try:
            from datetime import datetime
            ts = txn["timestamp"]
            if isinstance(ts, str):
                if ts.endswith('Z'): ts = ts[:-1]
                hour = datetime.fromisoformat(ts).hour
            elif hasattr(ts, "hour"):
                hour = ts.hour
        except Exception:
            pass
    hour = int(txn.get("hour", hour))
    is_night = 1 if hour < 6 or hour > 22 else 0

    # 7. device_change_frequency
    device_change_frequency = int(txn.get("device_changes", 0))

    return {
        "amount": amount,
        "amount_deviation": round(float(amount_deviation), 4),
        "location_change": float(location_change),
        "new_device": float(new_device),
        "merchant_risk": int(merchant_risk),
        "txn_velocity": int(txn_velocity),
        "is_night": int(is_night),
        "device_change_frequency": int(device_change_frequency),
    }


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_fraud(txn: dict, profile: dict) -> dict:
    """
    Run ML prediction on a transaction dict using existing profile.

    Returns:
        {
            "fraud_probability": float (0-1),
            "ml_risk_score": int (0-100),
            "features": dict
        }

    Raises RuntimeError on any failure — NEVER silently returns SAFE.
    """
    global model

    if model is None:
        raise RuntimeError("ML model is not loaded. Cannot predict — refusing to default to SAFE.")

    features_dict = extract_features(txn, profile)

    # Build feature vector in STRICT column order
    features_list = [features_dict[col] for col in FEATURE_COLS]

    # Validate feature count
    assert len(features_list) == model.n_features_in_, (
        f"Feature length mismatch: expected {model.n_features_in_}, got {len(features_list)}"
    )

    try:
        import pandas as pd
        X = pd.DataFrame([features_list], columns=FEATURE_COLS)
        prob = model.predict_proba(X)[0][1]
    except Exception as e:
        raise RuntimeError(f"ML prediction failed: {str(e)}")

    ml_score = int(prob * 100)

    # ── Debug visibility ──────────────────────────────────────────────────
    print("  [ML] Input:  %s" % features_list)
    print("  [ML] Prob:   %.4f  ->  score %d/100" % (prob, ml_score))

    return {
        "fraud_probability": float(prob),
        "ml_risk_score": ml_score,
        "features": features_dict,
    }


# ── Explainability ─────────────────────────────────────────────────────────────

def explain_prediction(features: dict) -> list:
    """
    Generate human-readable reasons based on feature values.
    These are injected into the API response for the dashboard.
    """
    reasons = []

    amount = features.get("amount", 0)
    if amount > 50000:
        reasons.append("High transaction amount (>₹50,000)")
    elif amount > 20000:
        reasons.append("Elevated transaction amount (>₹20,000)")

    loc_val = features.get("location_change", 0)
    if loc_val == 1.0:
        reasons.append("Transaction from a new/unusual location")
    elif loc_val == 0.5:
        reasons.append("Transaction from an unknown location")

    dev_val = features.get("new_device", 0)
    if dev_val == 1.0:
        reasons.append("New / unrecognized device")
    elif dev_val == 0.5:
        reasons.append("Device verification unavailable")

    if features.get("txn_velocity", 0) > 5:
        reasons.append("High transaction frequency")
    elif features.get("txn_velocity", 0) > 3:
        reasons.append("Elevated transaction frequency")

    if features.get("merchant_risk"):
        reasons.append("High-risk merchant category")

    if features.get("is_night"):
        reasons.append("Transaction at unusual hour (night)")

    if features.get("device_change_frequency", 0) > 2:
        reasons.append("Frequent device changes detected")

    if features.get("amount_deviation", 0) > 5:
        reasons.append("Amount significantly deviates from user baseline")

    return reasons
