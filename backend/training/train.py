# ============================================================
# UNIVERSAL CYBER ATTACK DETECTION
# TRAINING PIPELINE
#
# IALP + IFF + XGBoost
# KDDCup99
# ============================================================

import os
import sys
import joblib
import numpy as np

from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORTS
# ============================================================

from backend.preprocessing.dataset_adapter import (
    load_dataset,
    adapt_kdd99
)

from backend.preprocessing.cleaner import (
    clean_data
)

from backend.preprocessing.encoder import (
    DataEncoder
)

from backend.core.ialp import (
    IALP
)

from backend.core.iff import (
    IFF
)

from backend.core.xgboost_model import (
    XGBoostModel
)


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "train.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# TRAIN
# ============================================================

def train():

    print()
    print("=" * 60)
    print("       KDD99 CYBER ATTACK TRAINING")
    print("=" * 60)

    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    print()
    print("[1] Loading dataset...")

    if not os.path.exists(
        DATASET_PATH
    ):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = load_dataset(
        DATASET_PATH
    )

    # ========================================================
    # 2. ADAPT KDD99
    # ========================================================

    print()
    print("[2] Adapting KDDCup99...")

    X, y = adapt_kdd99(
        df
    )

    # ========================================================
    # 3. CLEAN
    # ========================================================

    print()
    print("[3] Cleaning data...")

    X = clean_data(
        X
    )

    # Ensure same number of samples
    X = X.reset_index(
        drop=True
    )

    y = y.reset_index(
        drop=True
    )

    if len(X) != len(y):

        raise ValueError(
            f"X/y size mismatch: "
            f"{len(X)} vs {len(y)}"
        )

    print(
        "Cleaned X shape:",
        X.shape
    )

    print(
        "Cleaned y shape:",
        y.shape
    )

    # ========================================================
    # 4. TRAIN TEST SPLIT
    # ========================================================

    print()
    print("[4] Creating train/test split...")

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # ========================================================
    # 5. ENCODER
    # ========================================================

    print()
    print("[5] Encoding features...")

    encoder = DataEncoder()

    X_train_encoded = (
        encoder.fit_transform(
            X_train
        )
    )

    X_test_encoded = (
        encoder.transform(
            X_test
        )
    )

    print(
        "Encoded training shape:",
        X_train_encoded.shape
    )

    print(
        "Encoded testing shape:",
        X_test_encoded.shape
    )

    # ========================================================
    # 6. IALP
    # ========================================================

    print()
    print("[6] Training IALP...")

    ialp = IALP()

    # Train IALP using NORMAL traffic only
    normal_mask = (
        y_train.values
        == "Normal"
    )

    X_normal = (
        X_train_encoded.loc[
            normal_mask
        ]
    )

    print(
        "Normal training records:",
        len(X_normal)
    )

    ialp.fit(
        X_normal
    )

    X_train_ialp = (
        ialp.transform(
            X_train_encoded
        )
    )

    X_test_ialp = (
        ialp.transform(
            X_test_encoded
        )
    )

    print(
        "IALP training shape:",
        X_train_ialp.shape
    )

    print(
        "IALP testing shape:",
        X_test_ialp.shape
    )

    print(
        "IALP information:",
        ialp.get_info()
    )

    # ========================================================
    # 7. IFF
    # ========================================================

    print()
    print("[7] Training IFF / Isolation Forest...")

    iff = IFF(
        n_estimators=150,
        contamination="auto",
        random_state=42
    )

    # IFF also learns NORMAL traffic
    X_normal_ialp = (
        ialp.transform(
            X_normal
        )
    )

    iff.fit(
        X_normal_ialp
    )

    # IFF predictions and scores
    train_iff_prediction = (
        iff.predict(
            X_train_ialp
        )
    )

    train_iff_score = (
        iff.score_samples(
            X_train_ialp
        )
    )

    test_iff_prediction = (
        iff.predict(
            X_test_ialp
        )
    )

    test_iff_score = (
        iff.score_samples(
            X_test_ialp
        )
    )

    # ========================================================
    # 8. FEATURE FUSION
    # ========================================================

    print()
    print("[8] Creating final feature representation...")

    X_train_final = np.column_stack(
        [
            X_train_ialp,
            train_iff_prediction,
            train_iff_score
        ]
    )

    X_test_final = np.column_stack(
        [
            X_test_ialp,
            test_iff_prediction,
            test_iff_score
        ]
    )

    print(
        "Final training shape:",
        X_train_final.shape
    )

    print(
        "Final testing shape:",
        X_test_final.shape
    )

    expected_features = (
        X_train_encoded.shape[1]
        + 2
        + 2
    )

    print(
        "Expected final features:",
        expected_features
    )

    if (
        X_train_final.shape[1]
        != expected_features
    ):

        raise ValueError(
            "Unexpected final feature count."
        )

    # ========================================================
    # 9. XGBOOST
    # ========================================================

    print()
    print("[9] Training XGBoost...")

    xgb_model = XGBoostModel()

    xgb_model.fit(
        X_train_final,
        y_train.values
    )

    print(
        "XGBoost training completed."
    )

    # ========================================================
    # 10. SAVE MODELS
    # ========================================================

    print()
    print("[10] Saving models...")

    encoder_path = os.path.join(
        MODEL_DIR,
        "encoder.pkl"
    )

    ialp_path = os.path.join(
        MODEL_DIR,
        "ialp.pkl"
    )

    iff_path = os.path.join(
        MODEL_DIR,
        "iff.pkl"
    )

    xgb_path = os.path.join(
        MODEL_DIR,
        "xgboost.pkl"
    )

    joblib.dump(
        encoder,
        encoder_path
    )

    joblib.dump(
        ialp,
        ialp_path
    )

    joblib.dump(
        iff,
        iff_path
    )

    joblib.dump(
        xgb_model,
        xgb_path
    )

    print()
    print("Models saved successfully:")

    print(
        encoder_path
    )

    print(
        ialp_path
    )

    print(
        iff_path
    )

    print(
        xgb_path
    )

    # ========================================================
    # 11. QUICK TEST
    # ========================================================

    print()
    print("[11] Running quick model test...")

    sample_features = X_test_final[
        :5
    ]

    predictions = (
        xgb_model.predict_with_confidence(
            sample_features
        )
    )

    for index, result in enumerate(
        predictions
    ):

        print(
            f"Sample {index + 1}: "
            f"{result['label']} "
            f"({result['confidence']:.2f}%)"
        )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 60)
    print("          TRAINING COMPLETED")
    print("=" * 60)

    print()
    print(
        "Pipeline:"
    )

    print(
        "KDDCup99"
        " -> Cleaning"
        " -> Encoding"
        " -> IALP"
        " -> IFF"
        " -> XGBoost"
    )

    print()
    print(
        "Models are ready for the API."
    )

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()