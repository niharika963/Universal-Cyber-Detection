import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from backend.preprocessing.dataset_adapter import (
    load_dataset,
    adapt_kdd99
)

from backend.preprocessing.cleaner import (
    clean_data
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "models"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "train.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# We do NOT analyze all 494,021 records through the web request.
# This keeps the dashboard responsive.

MAX_ANALYSIS_ROWS = 20000

RANDOM_STATE = 42


# ============================================================
# ATTACK CLASS ORDER
# ============================================================

CLASS_ORDER = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
]


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

def load_models():

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

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            "encoder.pkl was not found."
        )

    if not os.path.exists(ialp_path):
        raise FileNotFoundError(
            "ialp.pkl was not found."
        )

    if not os.path.exists(iff_path):
        raise FileNotFoundError(
            "iff.pkl was not found."
        )

    if not os.path.exists(xgb_path):
        raise FileNotFoundError(
            "xgboost.pkl was not found."
        )

    encoder = joblib.load(
        encoder_path
    )

    ialp_model = joblib.load(
        ialp_path
    )

    iff_model = joblib.load(
        iff_path
    )

    xgb_model = joblib.load(
        xgb_path
    )

    return (
        encoder,
        ialp_model,
        iff_model,
        xgb_model
    )


# ============================================================
# STRATIFIED SAMPLE
# ============================================================

def create_analysis_sample(
    X,
    y,
    max_rows=MAX_ANALYSIS_ROWS
):

    data = X.copy()

    data["_target_"] = y.values

    total_rows = len(data)

    if total_rows <= max_rows:

        data = data.sample(
            frac=1,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

        y_sample = data.pop(
            "_target_"
        )

        return (
            data,
            y_sample.reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Calculate samples per class
    # --------------------------------------------------------

    class_counts = (
        data["_target_"]
        .value_counts()
    )

    number_of_classes = len(
        class_counts
    )

    base_per_class = max(
        1,
        max_rows // number_of_classes
    )

    sampled_parts = []

    remaining = max_rows

    # --------------------------------------------------------
    # First take an equal-ish amount from every class.
    # This is important because U2R and R2L are rare.
    # --------------------------------------------------------

    for class_name in CLASS_ORDER:

        class_data = data[
            data["_target_"] == class_name
        ]

        if len(class_data) == 0:
            continue

        take = min(
            len(class_data),
            base_per_class
        )

        part = class_data.sample(
            n=take,
            random_state=RANDOM_STATE
        )

        sampled_parts.append(
            part
        )

        remaining -= take

    # --------------------------------------------------------
    # Fill remaining rows from unused records
    # --------------------------------------------------------

    if remaining > 0:

        already_sampled = pd.concat(
            sampled_parts,
            axis=0
        )

        remaining_data = data.drop(
            already_sampled.index,
            errors="ignore"
        )

        if len(remaining_data) > 0:

            take = min(
                remaining,
                len(remaining_data)
            )

            extra = remaining_data.sample(
                n=take,
                random_state=RANDOM_STATE
            )

            sampled_parts.append(
                extra
            )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    sample = pd.concat(
        sampled_parts,
        axis=0
    )

    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    y_sample = sample.pop(
        "_target_"
    )

    return (
        sample,
        y_sample.reset_index(drop=True)
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    X,
    encoder,
    ialp_model,
    iff_model
):

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    encoded = encoder.transform(
        X
    )

    if not isinstance(
        encoded,
        pd.DataFrame
    ):

        encoded = pd.DataFrame(
            encoded
        )

    # --------------------------------------------------------
    # IALP transformation
    # --------------------------------------------------------

    ialp_features = ialp_model.transform(
        encoded
    )

    if not isinstance(
        ialp_features,
        pd.DataFrame
    ):

        ialp_features = pd.DataFrame(
            ialp_features
        )

    ialp_features = (
        ialp_features
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # IFF
    # --------------------------------------------------------

    iff_prediction = iff_model.predict(
        ialp_features
    )

    iff_score = iff_model.decision_function(
        ialp_features
    )

    iff_prediction = np.asarray(
        iff_prediction
    ).reshape(-1)

    iff_score = np.asarray(
        iff_score
    ).reshape(-1)

    # --------------------------------------------------------
    # Feature Fusion
    # --------------------------------------------------------

    final_features = ialp_features.copy()

    final_features[
        "iff_prediction"
    ] = iff_prediction

    final_features[
        "iff_score"
    ] = iff_score

    return (
        final_features,
        iff_prediction,
        iff_score
    )


# ============================================================
# PREDICT
# ============================================================

def make_predictions(
    final_features,
    xgb_model
):

    results = xgb_model.predict_with_confidence(
        final_features
    )

    predictions = [
        str(item["label"])
        for item in results
    ]

    confidences = [
        float(item["confidence"])
        for item in results
    ]

    return (
        np.array(predictions),
        np.array(confidences)
    )


# ============================================================
# CLASS METRICS
# ============================================================

def calculate_class_metrics(
    y_true,
    y_pred
):

    result = {}

    for class_name in CLASS_ORDER:

        true_binary = (
            np.array(y_true)
            == class_name
        )

        pred_binary = (
            np.array(y_pred)
            == class_name
        )

        precision = precision_score(
            true_binary,
            pred_binary,
            zero_division=0
        )

        recall = recall_score(
            true_binary,
            pred_binary,
            zero_division=0
        )

        f1 = f1_score(
            true_binary,
            pred_binary,
            zero_division=0
        )

        support = int(
            true_binary.sum()
        )

        result[class_name] = {

            "precision": round(
                float(precision * 100),
                2
            ),

            "recall": round(
                float(recall * 100),
                2
            ),

            "f1": round(
                float(f1 * 100),
                2
            ),

            "support": support
        }

    return result


# ============================================================
# ROC DATA
# ============================================================

def calculate_roc(
    y_true,
    probabilities,
    class_names
):

    roc_result = {}

    for index, class_name in enumerate(
        class_names
    ):

        true_binary = (
            np.array(y_true)
            == class_name
        ).astype(int)

        # ----------------------------------------------------
        # ROC needs both positive and negative samples.
        # ----------------------------------------------------

        if len(np.unique(true_binary)) < 2:

            roc_result[class_name] = {

                "fpr": [],

                "tpr": [],

                "auc": None
            }

            continue

        try:

            scores = probabilities[
                :,
                index
            ]

            fpr, tpr, _ = roc_curve(
                true_binary,
                scores
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            roc_result[class_name] = {

                "fpr": [
                    float(x)
                    for x in fpr
                ],

                "tpr": [
                    float(x)
                    for x in tpr
                ],

                "auc": round(
                    float(roc_auc),
                    4
                )
            }

        except Exception:

            roc_result[class_name] = {

                "fpr": [],

                "tpr": [],

                "auc": None
            }

    return roc_result


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def run_full_analysis():

    print("\n")
    print("=" * 60)
    print("UNIVERSAL CYBER ATTACK ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            "datasets/train.csv was not found."
        )

    print("\n[1] Loading dataset...")

    df = load_dataset(
        DATASET_PATH
    )

    print(
        "Full dataset rows:",
        len(df)
    )

    # --------------------------------------------------------
    # Adapt KDD99
    # --------------------------------------------------------

    print("\n[2] Adapting KDDCup99...")

    X, y = adapt_kdd99(
        df
    )

    # --------------------------------------------------------
    # Cleaning
    # --------------------------------------------------------

    print("\n[3] Cleaning data...")

    X = clean_data(
        X
    )

    # --------------------------------------------------------
    # Create analysis sample
    # --------------------------------------------------------

    print(
        "\n[4] Creating stratified analysis sample..."
    )

    X_sample, y_sample = create_analysis_sample(
        X,
        y,
        MAX_ANALYSIS_ROWS
    )

    print(
        "Analysis rows:",
        len(X_sample)
    )

    print(
        "Analysis distribution:"
    )

    print(
        y_sample.value_counts()
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print(
        "\n[5] Loading trained models..."
    )

    (
        encoder,
        ialp_model,
        iff_model,
        xgb_model
    ) = load_models()

    print(
        "Encoder : Loaded"
    )

    print(
        "IALP    : Loaded"
    )

    print(
        "IFF     : Loaded"
    )

    print(
        "XGBoost : Loaded"
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    print(
        "\n[6] Running IALP + IFF..."
    )

    (
        final_features,
        iff_prediction,
        iff_score
    ) = prepare_features(
        X_sample,
        encoder,
        ialp_model,
        iff_model
    )

    print(
        "Final feature shape:",
        final_features.shape
    )

    # --------------------------------------------------------
    # XGBoost prediction
    # --------------------------------------------------------

    print(
        "\n[7] Running XGBoost..."
    )

    (
        y_pred,
        confidences
    ) = make_predictions(
        final_features,
        xgb_model
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = xgb_model.predict_proba(
        final_features
    )

    probabilities = np.asarray(
        probabilities
    )

    # --------------------------------------------------------
    # Determine model class order
    # --------------------------------------------------------

    if hasattr(
        xgb_model,
        "reverse_mapping"
    ):

        mapping = xgb_model.reverse_mapping

        class_names = [
            mapping[int(i)]
            for i in range(
                len(mapping)
            )
        ]

    else:

        class_names = CLASS_ORDER.copy()

    # Ensure standard display order
    ordered_classes = [
        cls
        for cls in CLASS_ORDER
        if cls in class_names
    ]

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_sample,
        y_pred
    )

    macro_precision = precision_score(
        y_sample,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_sample,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_sample,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_sample,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_sample,
        y_pred,
        labels=ordered_classes
    )

    matrix_list = [
        [
            int(value)
            for value in row
        ]
        for row in matrix
    ]

    # --------------------------------------------------------
    # Class performance
    # --------------------------------------------------------

    class_performance = (
        calculate_class_metrics(
            y_sample,
            y_pred
        )
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    roc_data = calculate_roc(
        y_sample,
        probabilities,
        class_names
    )

    # --------------------------------------------------------
    # IFF statistics
    # --------------------------------------------------------

    anomaly_count = int(
        np.sum(
            iff_prediction == -1
        )
    )

    normal_count = int(
        np.sum(
            iff_prediction == 1
        )
    )

    anomaly_percentage = (
        anomaly_count
        / len(iff_prediction)
        * 100
    )

    # --------------------------------------------------------
    # Model comparison
    #
    # We use the already-trained proposed model here.
    # We do not retrain SVM/RF/GBM during a web request.
    # --------------------------------------------------------

    comparison = [

        {
            "algorithm":
                "IALP + IFF + XGBoost",

            "accuracy":
                round(
                    float(accuracy * 100),
                    2
                ),

            "precision":
                round(
                    float(
                        macro_precision * 100
                    ),
                    2
                ),

            "recall":
                round(
                    float(
                        macro_recall * 100
                    ),
                    2
                ),

            "f1":
                round(
                    float(
                        macro_f1 * 100
                    ),
                    2
                ),

            "weighted_f1":
                round(
                    float(
                        weighted_f1 * 100
                    ),
                    2
                )
        }

    ]

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "success": True,

        "analysis_type":
            "Stratified hold-out analysis sample",

        "dataset": {

            "name":
                "KDDCup99",

            "total_records":
                int(len(df)),

            "analysis_records":
                int(len(X_sample)),

            "features":
                41,

            "classes":
                ordered_classes
        },

        "proposed_model": {

            "name":
                "IALP + IFF + XGBoost",

            "metrics": {

                "accuracy":
                    round(
                        float(
                            accuracy * 100
                        ),
                        2
                    ),

                "precision":
                    round(
                        float(
                            macro_precision * 100
                        ),
                        2
                    ),

                "recall":
                    round(
                        float(
                            macro_recall * 100
                        ),
                        2
                    ),

                "f1":
                    round(
                        float(
                            macro_f1 * 100
                        ),
                        2
                    ),

                "weighted_f1":
                    round(
                        float(
                            weighted_f1 * 100
                        ),
                        2
                    )
            },

            "confusion_matrix": {

                "labels":
                    ordered_classes,

                "matrix":
                    matrix_list
            },

            "class_metrics":
                class_performance,

            "roc":
                roc_data
        },

        "classes":
            ordered_classes,

        "comparison":
            comparison,

        "algorithm_comparison":
            comparison,

        "confusion_matrix":
            matrix_list,

        "class_performance":
            class_performance,

        "roc":
            roc_data,

        "iff": {

            "normal_records":
                normal_count,

            "anomalous_records":
                anomaly_count,

            "anomaly_percentage":
                round(
                    float(
                        anomaly_percentage
                    ),
                    2
                )
        },

        "message":
            "Analysis completed successfully."
    }

    print("\n")
    print("=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)

    print(
        "Accuracy:",
        round(
            accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Macro F1:",
        round(
            macro_f1 * 100,
            2
        ),
        "%"
    )

    print(
        "Analysis rows:",
        len(X_sample)
    )

    print("=" * 60)

    return result