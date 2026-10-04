import gc
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Render Free has a 512 MB RAM limit.
# Keep interactive evaluation deliberately small.
MAX_ROWS = 1500


# ============================================================
# POSSIBLE LABEL COLUMNS
# ============================================================

LABEL_COLUMNS = [
    "label",
    "class",
    "target",
    "attack",
    "attack_type",
    "connection_type",
    "category",
    "output",
]


# ============================================================
# KDD99 ATTACK GROUPING
# ============================================================

KDD_ATTACK_MAPPING = {

    "normal": "Normal",

    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",

    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",

    # R2L
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",

    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
}


# ============================================================
# FIND LABEL COLUMN
# ============================================================

def find_label_column(df):

    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in df.columns
    }

    for candidate in LABEL_COLUMNS:

        if candidate in normalized:
            return normalized[candidate]

    return None


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def normalize_columns(df):

    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        for column in df.columns
    ]

    return df


# ============================================================
# NORMALIZE LABELS
# ============================================================

def normalize_labels(y):

    y = y.astype(str).str.strip()

    result = []

    for value in y:

        key = value.lower().rstrip(".")

        result.append(
            KDD_ATTACK_MAPPING.get(
                key,
                value
            )
        )

    return pd.Series(
        result,
        index=y.index
    )


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(df):

    df = normalize_columns(df)

    original_records = len(df)

    label_column = find_label_column(df)

    if label_column is None:

        raise ValueError(
            "No label column found. "
            "Use one of: label, class, target, "
            "attack, attack_type, connection_type "
            "or category."
        )

    # --------------------------------------------------------
    # REMOVE EMPTY / DUPLICATE ROWS
    # --------------------------------------------------------

    df = df.dropna(
        how="all"
    )

    df = df.drop_duplicates()

    if len(df) < 20:

        raise ValueError(
            "Dataset must contain at least 20 records."
        )

    # --------------------------------------------------------
    # LIMIT DATASET SIZE
    # --------------------------------------------------------

    if len(df) > MAX_ROWS:

        print(
            f"Dataset contains {len(df)} records."
        )

        print(
            f"Limiting benchmark to {MAX_ROWS} records."
        )

        try:

            df, _ = train_test_split(
                df,
                train_size=MAX_ROWS,
                random_state=RANDOM_STATE,
                stratify=df[label_column]
            )

        except ValueError:

            df = df.sample(
                n=MAX_ROWS,
                random_state=RANDOM_STATE
            )

        df = df.reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # NORMALIZE LABELS
    # --------------------------------------------------------

    y = normalize_labels(
        df[label_column]
    )

    # --------------------------------------------------------
    # REMOVE CLASSES WITH LESS THAN 2 RECORDS
    # --------------------------------------------------------

    class_counts = y.value_counts()

    rare_classes = class_counts[
        class_counts < 2
    ].index.tolist()

    if rare_classes:

        print(
            "Rare classes excluded:",
            [
                str(value)
                for value in rare_classes
            ]
        )

        valid_mask = ~y.isin(
            rare_classes
        )

        X = (
            df.drop(
                columns=[label_column]
            )
            .loc[valid_mask]
            .reset_index(
                drop=True
            )
        )

        y = (
            y.loc[valid_mask]
            .reset_index(
                drop=True
            )
        )

    else:

        X = df.drop(
            columns=[label_column]
        ).copy()

        y = y.reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # REMOVE CONSTANT COLUMNS
    # --------------------------------------------------------

    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_columns:

        print(
            "Constant columns removed:",
            constant_columns
        )

        X = X.drop(
            columns=constant_columns
        )

    # --------------------------------------------------------
    # CONVERT NUMERIC STRINGS
    # --------------------------------------------------------

    for column in X.columns:

        if X[column].dtype == "object":

            converted = pd.to_numeric(
                X[column],
                errors="coerce"
            )

            numeric_ratio = (
                converted.notna().mean()
            )

            if numeric_ratio >= 0.95:

                X[column] = converted

    return (
        X,
        y,
        label_column,
        original_records,
        [
            str(value)
            for value in rare_classes
        ]
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X):

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "category",
            "string"
        ]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=[
            "number",
            "bool"
        ]
    ).columns.tolist()

    # --------------------------------------------------------
    # NUMERIC PIPELINE
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # CATEGORICAL PIPELINE
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # sparse_output=True prevents huge dense matrices.
    #

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    transformers = []

    if numerical_columns:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numerical_columns
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    if not transformers:

        raise ValueError(
            "No usable numerical or categorical "
            "features found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# EVALUATE STANDARD MODEL
# ============================================================

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {

        "algorithm":
            name,

        "accuracy":
            round(
                accuracy * 100,
                4
            ),

        "precision":
            round(
                precision * 100,
                4
            ),

        "recall":
            round(
                recall * 100,
                4
            ),

        "f1":
            round(
                f1 * 100,
                4
            ),

        "weighted_f1":
            round(
                weighted_f1 * 100,
                4
            ),

        "training_time":
            round(
                elapsed,
                3
            ),

        "predictions":
            predictions

    }, model


# ============================================================
# PROPOSED MODEL
#
# IALP + IFF + XGBoost
# ============================================================

def evaluate_proposed_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # CONVERT TO DENSE ONLY FOR PROPOSED MODEL
    # --------------------------------------------------------
    #
    # Baseline models remain sparse.
    # The benchmark is capped at 1500 rows, so this conversion
    # remains bounded.
    #

    if hasattr(
        X_train,
        "toarray"
    ):

        X_train_dense = (
            X_train
            .toarray()
            .astype(
                np.float32,
                copy=False
            )
        )

        X_test_dense = (
            X_test
            .toarray()
            .astype(
                np.float32,
                copy=False
            )
        )

    else:

        X_train_dense = np.asarray(
            X_train,
            dtype=np.float32
        )

        X_test_dense = np.asarray(
            X_test,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # IALP-INSPIRED FEATURE NORMALIZATION
    # --------------------------------------------------------

    train_mean = np.mean(
        X_train_dense,
        axis=0,
        dtype=np.float32
    )

    train_std = np.std(
        X_train_dense,
        axis=0,
        dtype=np.float32
    )

    train_std = np.where(
        train_std < 1e-8,
        1.0,
        train_std
    ).astype(
        np.float32
    )

    X_train_ialp = (
        (
            X_train_dense
            - train_mean
        )
        / train_std
    ).astype(
        np.float32,
        copy=False
    )

    X_test_ialp = (
        (
            X_test_dense
            - train_mean
        )
        / train_std
    ).astype(
        np.float32,
        copy=False
    )

    del X_train_dense
    del X_test_dense
    del train_mean
    del train_std

    gc.collect()

    # --------------------------------------------------------
    # IFF / ISOLATION FOREST
    # --------------------------------------------------------

    iff = IsolationForest(
        n_estimators=30,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    iff.fit(
        X_train_ialp
    )

    train_anomaly_score = (
        iff.decision_function(
            X_train_ialp
        )
        .reshape(
            -1,
            1
        )
        .astype(
            np.float32
        )
    )

    test_anomaly_score = (
        iff.decision_function(
            X_test_ialp
        )
        .reshape(
            -1,
            1
        )
        .astype(
            np.float32
        )
    )

    del iff

    gc.collect()

    # --------------------------------------------------------
    # COMBINE FEATURES
    # --------------------------------------------------------

    X_train_final = np.hstack(
        [
            X_train_ialp,
            train_anomaly_score
        ]
    ).astype(
        np.float32,
        copy=False
    )

    X_test_final = np.hstack(
        [
            X_test_ialp,
            test_anomaly_score
        ]
    ).astype(
        np.float32,
        copy=False
    )

    del X_train_ialp
    del X_test_ialp
    del train_anomaly_score
    del test_anomaly_score

    gc.collect()

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb = XGBClassifier(

        n_estimators=50,

        max_depth=4,

        learning_rate=0.1,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=RANDOM_STATE,

        n_jobs=1,

        eval_metric="mlogloss"
    )

    xgb.fit(
        X_train_final,
        y_train
    )

    predictions = xgb.predict(
        X_test_final
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # RELEASE PROPOSED MODEL
    # --------------------------------------------------------

    del xgb
    del X_train_final
    del X_test_final

    gc.collect()

    return {

        "algorithm":
            "IALP + IFF + XGBoost",

        "accuracy":
            round(
                accuracy * 100,
                4
            ),

        "precision":
            round(
                precision * 100,
                4
            ),

        "recall":
            round(
                recall * 100,
                4
            ),

        "f1":
            round(
                f1 * 100,
                4
            ),

        "weighted_f1":
            round(
                weighted_f1 * 100,
                4
            ),

        "training_time":
            round(
                elapsed,
                3
            ),

        "predictions":
            predictions
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def build_confusion_matrix(
    y_test,
    predictions,
    label_encoder
):

    matrix = confusion_matrix(

        y_test,

        predictions,

        labels=range(
            len(
                label_encoder.classes_
            )
        )
    )

    return {

        "labels": [
            str(label)
            for label
            in label_encoder.classes_
        ],

        "matrix":
            matrix.tolist()
    }


# ============================================================
# CLASS PERFORMANCE
# ============================================================

def build_class_performance(
    y_test,
    predictions,
    label_encoder
):

    labels = range(
        len(
            label_encoder.classes_
        )
    )

    precision_values = precision_score(

        y_test,

        predictions,

        average=None,

        labels=labels,

        zero_division=0
    )

    recall_values = recall_score(

        y_test,

        predictions,

        average=None,

        labels=labels,

        zero_division=0
    )

    f1_values = f1_score(

        y_test,

        predictions,

        average=None,

        labels=labels,

        zero_division=0
    )

    support_values = np.bincount(

        y_test,

        minlength=len(
            label_encoder.classes_
        )
    )

    result = {}

    for index, class_name in enumerate(
        label_encoder.classes_
    ):

        result[
            str(class_name)
        ] = {

            "precision":
                round(
                    float(
                        precision_values[index]
                        * 100
                    ),
                    4
                ),

            "recall":
                round(
                    float(
                        recall_values[index]
                        * 100
                    ),
                    4
                ),

            "f1":
                round(
                    float(
                        f1_values[index]
                        * 100
                    ),
                    4
                ),

            "support":
                int(
                    support_values[index]
                )
        }

    return result


# ============================================================
# ROC CURVE
# ============================================================

def build_roc_data(
    model,
    X_test,
    y_test,
    label_encoder
):

    print(
        "Calculating ROC curve..."
    )

    if not hasattr(
        model,
        "predict_proba"
    ):

        print(
            "Model does not support predict_proba."
        )

        return []

    try:

        probabilities = model.predict_proba(
            X_test
        )

    except Exception as error:

        print(
            "ROC probability error:",
            error
        )

        return []

    roc_results = []

    number_of_classes = len(
        label_encoder.classes_
    )

    for class_id in range(
        number_of_classes
    ):

        binary_actual = (
            y_test == class_id
        ).astype(
            int
        )

        # ROC cannot be calculated if
        # test data contains only one
        # state for this class.

        if len(
            np.unique(
                binary_actual
            )
        ) < 2:

            class_name = (
                label_encoder
                .inverse_transform(
                    [class_id]
                )[0]
            )

            print(
                "Skipping ROC class:",
                class_name
            )

            continue

        if class_id >= probabilities.shape[1]:

            continue

        fpr, tpr, _ = roc_curve(

            binary_actual,

            probabilities[
                :,
                class_id
            ]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        class_name = (
            label_encoder
            .inverse_transform(
                [class_id]
            )[0]
        )

        roc_results.append(
            {

                "class_name":
                    str(class_name),

                "class":
                    str(class_name),

                "fpr": [
                    round(
                        float(value),
                        6
                    )
                    for value
                    in fpr
                ],

                "tpr": [
                    round(
                        float(value),
                        6
                    )
                    for value
                    in tpr
                ],

                "auc":
                    round(
                        float(roc_auc),
                        6
                    )
            }
        )

    print(
        "ROC curves generated:",
        len(roc_results)
    )

    del probabilities

    gc.collect()

    return roc_results


# ============================================================
# MAIN DATASET BENCHMARK
# ============================================================

def run_dataset_benchmark(df):

    print(
        "========================================"
    )

    print(
        "STARTING DATASET BENCHMARK"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # PREPARE DATASET
    # --------------------------------------------------------

    (
        X,
        y,
        label_column,
        original_records,
        excluded_classes
    ) = prepare_dataset(
        df
    )

    print(
        "Records used:",
        len(X)
    )

    print(
        "Features:",
        X.shape[1]
    )

    # --------------------------------------------------------
    # LABEL ENCODING
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y
    )

    if len(
        np.unique(
            y_encoded
        )
    ) < 2:

        raise ValueError(
            "Dataset must contain at least "
            "two classes after preprocessing."
        )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,

        y_encoded,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y_encoded
    )

    print(
        "Training records:",
        len(X_train)
    )

    print(
        "Testing records:",
        len(X_test)
    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        X_train
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # KEEP BASELINE MATRIX SPARSE
    # --------------------------------------------------------

    if hasattr(
        X_train_processed,
        "tocsr"
    ):

        X_train_processed = (
            X_train_processed
            .tocsr()
            .astype(
                np.float32
            )
        )

        X_test_processed = (
            X_test_processed
            .tocsr()
            .astype(
                np.float32
            )
        )

    else:

        X_train_processed = np.asarray(
            X_train_processed,
            dtype=np.float32
        )

        X_test_processed = np.asarray(
            X_test_processed,
            dtype=np.float32
        )

    print(
        "Processed features:",
        X_train_processed.shape[1]
    )

    # --------------------------------------------------------
    # RELEASE RAW DATAFRAME / PREPROCESSOR
    # --------------------------------------------------------

    del df
    del preprocessor

    gc.collect()

    # --------------------------------------------------------
    # BASELINE MODELS
    # --------------------------------------------------------

    models = {

        "Decision Tree":

            DecisionTreeClassifier(

                random_state=
                    RANDOM_STATE,

                max_depth=10
            ),

        "Random Forest":

            RandomForestClassifier(

                n_estimators=30,

                random_state=
                    RANDOM_STATE,

                n_jobs=1,

                max_depth=10
            ),

        "XGBoost":

            XGBClassifier(

                n_estimators=50,

                max_depth=4,

                learning_rate=0.1,

                subsample=0.8,

                colsample_bytree=0.8,

                random_state=
                    RANDOM_STATE,

                n_jobs=1,

                eval_metric=
                    "mlogloss"
            ),

        "CatBoost":

            CatBoostClassifier(

                iterations=50,

                depth=4,

                learning_rate=0.1,

                verbose=False,

                random_seed=
                    RANDOM_STATE,

                thread_count=1
            ),

        "LightGBM":

            LGBMClassifier(

                n_estimators=50,

                learning_rate=0.1,

                max_depth=8,

                random_state=
                    RANDOM_STATE,

                n_jobs=1,

                verbosity=-1
            )
    }

    comparison = []

    xgb_baseline_model = None

    # --------------------------------------------------------
    # TRAIN BASELINES ONE AT A TIME
    # --------------------------------------------------------

    for name, model in models.items():

        print(
            "Training:",
            name
        )

        (
            result,
            fitted_model
        ) = evaluate_model(

            name,

            model,

            X_train_processed,

            X_test_processed,

            y_train,

            y_test
        )

        predictions = result.pop(
            "predictions"
        )

        comparison.append(
            result
        )

        # ----------------------------------------------------
        # KEEP ONLY BASELINE XGBOOST
        # ----------------------------------------------------
        #
        # XGBoost is retained temporarily for ROC.
        # All other models are released immediately.
        #

        if name == "XGBoost":

            xgb_baseline_model = (
                fitted_model
            )

        else:

            del fitted_model

        del model
        del predictions

        gc.collect()

    del models

    gc.collect()

    # --------------------------------------------------------
    # ROC FOR BASELINE XGBOOST
    # --------------------------------------------------------
    #
    # Do this BEFORE the proposed model.
    # This prevents two XGBoost models from staying in memory
    # at the same time.
    #

    if xgb_baseline_model is not None:

        roc_data = build_roc_data(

            xgb_baseline_model,

            X_test_processed,

            y_test,

            label_encoder
        )

        del xgb_baseline_model

        xgb_baseline_model = None

        gc.collect()

    else:

        roc_data = []

    # --------------------------------------------------------
    # PROPOSED MODEL
    # --------------------------------------------------------

    print(
        "Training: IALP + IFF + XGBoost"
    )

    proposed_result = (
        evaluate_proposed_model(

            X_train_processed,

            X_test_processed,

            y_train,

            y_test
        )
    )

    proposed_predictions = (
        proposed_result.pop(
            "predictions"
        )
    )

    comparison.append(
        proposed_result
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    proposed_confusion = (
        build_confusion_matrix(

            y_test,

            proposed_predictions,

            label_encoder
        )
    )

    # --------------------------------------------------------
    # CLASS PERFORMANCE
    # --------------------------------------------------------

    class_performance = (
        build_class_performance(

            y_test,

            proposed_predictions,

            label_encoder
        )
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    class_distribution = {}

    (
        unique_classes,
        class_counts
    ) = np.unique(

        y_encoded,

        return_counts=True
    )

    for class_id, count in zip(
        unique_classes,
        class_counts
    ):

        class_name = (
            label_encoder
            .inverse_transform(
                [class_id]
            )[0]
        )

        class_distribution[
            str(class_name)
        ] = int(
            count
        )

    # --------------------------------------------------------
    # SAVE SMALL VALUES BEFORE CLEANUP
    # --------------------------------------------------------

    records_used = int(
        len(X)
    )

    feature_count = int(
        X.shape[1]
    )

    train_records = int(
        len(X_train)
    )

    test_records = int(
        len(X_test)
    )

    class_names = [
        str(value)
        for value
        in label_encoder.classes_
    ]

    class_count = int(
        len(
            label_encoder.classes_
        )
    )

    # --------------------------------------------------------
    # COMPLETION LOG
    # --------------------------------------------------------

    print(
        "========================================"
    )

    print(
        "DATASET BENCHMARK COMPLETED"
    )

    print(
        "ROC DATA POINTS:",
        len(roc_data)
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # RELEASE LARGE TEMPORARY OBJECTS
    # --------------------------------------------------------

    del X_train_processed
    del X_test_processed

    del X_train
    del X_test

    del y_train
    del y_test

    del X
    del y
    del y_encoded

    del proposed_predictions
    del label_encoder

    gc.collect()

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {

        "dataset": {

            "records_used":
                records_used,

            "original_records":
                int(
                    original_records
                ),

            "excluded_rare_classes":
                excluded_classes,

            "features":
                feature_count,

            "classes":
                class_count,

            "label_column":
                str(
                    label_column
                ),

            "class_names":
                class_names,

            "class_distribution":
                class_distribution
        },

        "split": {

            "train_records":
                train_records,

            "test_records":
                test_records,

            "test_size":
                0.20,

            "random_state":
                RANDOM_STATE
        },

        "comparison":
            comparison,

        "confusion_matrix":
            proposed_confusion,

        "class_performance":
            class_performance,

        "roc":
            roc_data,

        "roc_data":
            roc_data
    }