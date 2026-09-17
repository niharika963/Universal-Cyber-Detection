import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# ==========================================
# IMPORT PROJECT MODULES
# ==========================================

from backend.preprocessing.dataset_adapter import (
    load_dataset,
    adapt_kdd99
)

from backend.preprocessing.cleaner import clean_data


# ==========================================
# PATHS
# ==========================================

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "datasets",
    "train.csv"
)

MODELS_PATH = os.path.join(
    PROJECT_DIR,
    "backend",
    "models"
)


# ==========================================
# START
# ==========================================

print("=" * 60)
print("       CYBER ATTACK DETECTION EVALUATION")
print("=" * 60)


# ==========================================
# LOAD DATASET
# ==========================================

df = load_dataset(DATASET_PATH)

X, y = adapt_kdd99(df)

X = clean_data(X)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# LOAD SAVED MODELS
# ==========================================

print("\nLoading trained models...")

encoder = joblib.load(
    os.path.join(
        MODELS_PATH,
        "encoder.pkl"
    )
)

ialp = joblib.load(
    os.path.join(
        MODELS_PATH,
        "ialp.pkl"
    )
)

iff = joblib.load(
    os.path.join(
        MODELS_PATH,
        "iff.pkl"
    )
)

xgb = joblib.load(
    os.path.join(
        MODELS_PATH,
        "xgboost.pkl"
    )
)

print("Models loaded successfully.")


# ==========================================
# ENCODING
# ==========================================

print("\nRunning preprocessing...")

X_test_encoded = encoder.transform(X_test)

print(
    "Encoded feature shape:",
    X_test_encoded.shape
)


# ==========================================
# IALP TRANSFORMATION
# ==========================================

X_test_ialp = ialp.transform(
    X_test_encoded
)

print(
    "IALP feature shape   :",
    X_test_ialp.shape
)


# ==========================================
# IFF / ISOLATION FOREST
# ==========================================

iff_prediction = iff.predict(
    X_test_ialp
)

iff_score = iff.anomaly_score(
    X_test_ialp
)

print(
    "IFF prediction shape :",
    iff_prediction.shape
)

print(
    "IFF score shape      :",
    iff_score.shape
)


# ==========================================
# FINAL XGBOOST FEATURES
# ==========================================
#
# IMPORTANT:
#
# The trained XGBoost model expects 121
# features.
#
# IALP output = 119
# IFF prediction = 1
# IFF score = 1
#
# Total = 121
#
# Do NOT add X_test_encoded here again.
# ==========================================

X_test_final = np.hstack(
    [
        X_test_ialp,
        iff_prediction.reshape(-1, 1),
        iff_score.reshape(-1, 1)
    ]
)

print(
    "Final feature shape  :",
    X_test_final.shape
)


# ==========================================
# FEATURE COUNT CHECK
# ==========================================

EXPECTED_FEATURES = 121

if X_test_final.shape[1] != EXPECTED_FEATURES:

    raise ValueError(
        f"Feature shape mismatch before XGBoost: "
        f"expected {EXPECTED_FEATURES}, "
        f"got {X_test_final.shape[1]}"
    )


# ==========================================
# XGBOOST PREDICTION
# ==========================================

print("\nRunning XGBoost prediction...")

predictions = xgb.predict(
    X_test_final
)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)


# ==========================================
# RESULTS
# ==========================================

print("\n" + "=" * 60)
print("                    RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

labels = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
]

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print()

# Header
print(
    f"{'Actual / Predicted':<20}"
    + "".join(
        f"{label:>10}"
        for label in labels
    )
)

# Rows
for label, row in zip(labels, cm):

    print(
        f"{label:<20}"
        + "".join(
            f"{value:>10}"
            for value in row
        )
    )


# ==========================================
# PER-CLASS SUMMARY
# ==========================================

report = classification_report(
    y_test,
    predictions,
    labels=labels,
    output_dict=True,
    zero_division=0
)

print("\nPer-Class Performance:")
print()

for label in labels:

    print(
        f"{label:<10}"
        f"Precision: {report[label]['precision']:.4f}   "
        f"Recall: {report[label]['recall']:.4f}   "
        f"F1: {report[label]['f1-score']:.4f}"
    )


# ==========================================
# MACRO / WEIGHTED F1
# ==========================================

print("\nOverall Metrics:")

print(
    f"Accuracy       : {accuracy:.4f}"
)

print(
    f"Macro Precision: {report['macro avg']['precision']:.4f}"
)

print(
    f"Macro Recall   : {report['macro avg']['recall']:.4f}"
)

print(
    f"Macro F1       : {report['macro avg']['f1-score']:.4f}"
)

print(
    f"Weighted F1    : {report['weighted avg']['f1-score']:.4f}"
)


# ==========================================
# FINISHED
# ==========================================

print("\n" + "=" * 60)
print("Evaluation completed successfully.")
print("=" * 60)