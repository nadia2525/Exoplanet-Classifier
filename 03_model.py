"""
Train and compare classifiers for exoplanet confirmation.

Run: python 03_model.py
Output: trained model saved to models/best_model.joblib,
        metrics printed to console, ROC curve saved to figures/
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, RocCurveDisplay,
)

DATA_PATH = "data/koi_clean.csv"
MODEL_DIR = "models"
FIG_DIR = "figures"

FEATURE_COLS = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr",
    "koi_steff", "koi_slogg", "koi_srad",
]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def prep_data(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["False Positive", "Confirmed"]))
    print(f"ROC-AUC: {auc:.3f}")
    return auc, probs


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = prep_data(df)

    log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    auc_lr, probs_lr = evaluate("Logistic Regression", log_reg, X_test, y_test)

    rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
    auc_rf, probs_rf = evaluate("Random Forest", rf, X_test, y_test)

    # Feature importance from the random forest
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)
    print("\nFeature importances (Random Forest):")
    print(importances.round(3))

    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Feature importance for exoplanet confirmation")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
    plt.close()

    # ROC curve comparison
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, probs_lr, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test, probs_rf, name="Random Forest", ax=ax)
    plt.title("ROC curve comparison")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # Save the better model + scaler for the Streamlit app
    best_model, best_name = (rf, "random_forest") if auc_rf >= auc_lr else (log_reg, "logistic_regression")
    joblib.dump({"model": best_model, "scaler": scaler, "features": FEATURE_COLS},
                f"{MODEL_DIR}/best_model.joblib")
    print(f"\nSaved best model ({best_name}) to {MODEL_DIR}/best_model.joblib")
