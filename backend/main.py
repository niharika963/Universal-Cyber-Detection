from pathlib import Path
from io import BytesIO
import traceback

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .preprocessing.dataset_adapter import (
    KDD99_FEATURES,
    get_dataset_info,
    is_kdd99_compatible,
)

from .training.dataset_benchmark import run_dataset_benchmark

try:
    from .model_analysis import run_full_analysis
except Exception:
    run_full_analysis = None


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

FRONTEND_DIR = PROJECT_DIR / "frontend"
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = PROJECT_DIR / "datasets"

IALP_PATH = MODEL_DIR / "ialp.pkl"
IFF_PATH = MODEL_DIR / "iff.pkl"
XGBOOST_PATH = MODEL_DIR / "xgboost.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Universal Cyber Attack Detection System",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL MODELS
# ============================================================

ialp_model = None
iff_model = None
xgb_model = None
encoder = None

model_load_error = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():

    global ialp_model
    global iff_model
    global xgb_model
    global encoder
    global model_load_error

    model_load_error = None

    try:

        if not IALP_PATH.exists():
            raise FileNotFoundError(
                f"IALP model not found: {IALP_PATH}"
            )

        if not IFF_PATH.exists():
            raise FileNotFoundError(
                f"IFF model not found: {IFF_PATH}"
            )

        if not XGBOOST_PATH.exists():
            raise FileNotFoundError(
                f"XGBoost model not found: {XGBOOST_PATH}"
            )

        if not ENCODER_PATH.exists():
            raise FileNotFoundError(
                f"Encoder model not found: {ENCODER_PATH}"
            )

        ialp_model = joblib.load(IALP_PATH)
        iff_model = joblib.load(IFF_PATH)
        xgb_model = joblib.load(XGBOOST_PATH)
        encoder = joblib.load(ENCODER_PATH)

        print()
        print("=" * 60)
        print("CYBER DETECTION MODELS LOADED")
        print("=" * 60)
        print("IALP       : Loaded")
        print("IFF        : Loaded")
        print("XGBoost    : Loaded")
        print("Encoder    : Loaded")
        print("=" * 60)
        print()

    except Exception as error:

        model_load_error = str(error)

        print()
        print("=" * 60)
        print("MODEL LOADING ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)
        print()

        ialp_model = None
        iff_model = None
        xgb_model = None
        encoder = None


load_models()


# ============================================================
# HELPERS
# ============================================================

def convert_to_dataframe(value):

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, pd.Series):
        return value.to_frame()

    if hasattr(value, "toarray"):
        value = value.toarray()

    array = np.asarray(value)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    return pd.DataFrame(array)


def convert_to_numpy(value):

    if isinstance(value, pd.DataFrame):
        return value.to_numpy(dtype=float)

    if hasattr(value, "toarray"):
        value = value.toarray()

    return np.asarray(
        value,
        dtype=float
    )


def check_models():

    if model_load_error:

        raise HTTPException(
            status_code=503,
            detail={
                "message": "Models are not loaded.",
                "error": model_load_error,
            },
        )


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def root():

    index_file = FRONTEND_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {
        "status": "online",
        "message": "Universal Cyber Attack Detection API",
    }


@app.get("/style.css")
def style():

    css_file = FRONTEND_DIR / "style.css"

    if not css_file.exists():

        raise HTTPException(
            status_code=404,
            detail="style.css not found.",
        )

    return FileResponse(
        css_file,
        media_type="text/css",
    )


@app.get("/app.js")
def javascript():

    js_file = FRONTEND_DIR / "app.js"

    if not js_file.exists():

        raise HTTPException(
            status_code=404,
            detail="app.js not found.",
        )

    return FileResponse(
        js_file,
        media_type="application/javascript",
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
        if (
            ialp_model is not None
            and iff_model is not None
            and xgb_model is not None
            and encoder is not None
        )
        else "degraded",

        "ialp_loaded":
            ialp_model is not None,

        "iff_loaded":
            iff_model is not None,

        "xgboost_loaded":
            xgb_model is not None,

        "encoder_loaded":
            encoder is not None,

        "model_load_error":
            model_load_error,
    }


# ============================================================
# INSPECT DATASET
# ============================================================

@app.post("/inspect-dataset")
async def inspect_dataset(
    file: UploadFile = File(...)
):

    try:

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file selected.",
            )

        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a CSV file.",
            )

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        dataframe = pd.read_csv(
            BytesIO(contents)
        )

        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV contains no records.",
            )

        information = get_dataset_info(
            dataframe
        )

        compatible = is_kdd99_compatible(
            dataframe
        )

        label_column = information.get(
            "label_column"
        )

        class_distribution = {}

        if (
            label_column
            and
            label_column in dataframe.columns
        ):

            counts = (
                dataframe[label_column]
                .astype(str)
                .value_counts()
                .to_dict()
            )

            class_distribution = {
                str(key): int(value)
                for key, value in counts.items()
            }

        return {

            "success": True,

            "filename":
                file.filename,

            "dataset":
                information,

            "compatible_with_current_model":
                bool(compatible),

            "label_column":
                label_column,

            "feature_count":
                len(dataframe.columns)
                -
                (
                    1
                    if label_column
                    else 0
                ),

            "row_count":
                len(dataframe),

            "columns":
                dataframe.columns.tolist(),

            "class_distribution":
                class_distribution,

            "message":
                (
                    "Dataset detected successfully."
                ),
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=400,
            detail=f"Dataset inspection failed: {error}",
        )


# ============================================================
# SINGLE RECORD PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    record: dict
):

    check_models()

    if not isinstance(record, dict):

        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object.",
        )

    missing_features = [
        feature
        for feature in KDD99_FEATURES
        if feature not in record
    ]

    if missing_features:

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Required features are missing.",

                "missing_features":
                    missing_features,
            },
        )

    try:

        input_data = {
            feature:
                record.get(feature)
            for feature in KDD99_FEATURES
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=KDD99_FEATURES,
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=f"Could not create input record: {error}",
        )


    # --------------------------------------------------------
    # FEATURE TYPES
    # --------------------------------------------------------

    categorical_features = [
        "protocol_type",
        "service",
        "flag",
    ]

    numeric_features = [
        feature
        for feature in KDD99_FEATURES
        if feature not in categorical_features
    ]


    # --------------------------------------------------------
    # FIX NUMERIC DATA
    # --------------------------------------------------------

    invalid_numeric_features = []

    for feature in numeric_features:

        input_df[feature] = pd.to_numeric(
            input_df[feature],
            errors="coerce"
        )

        value = input_df[feature].iloc[0]

        if (
            pd.isna(value)
            or
            not np.isfinite(float(value))
        ):

            invalid_numeric_features.append(
                feature
            )

        else:

            input_df[feature] = (
                input_df[feature]
                .astype(float)
            )


    if invalid_numeric_features:

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Invalid numerical values found.",

                "invalid_features":
                    invalid_numeric_features,
            },
        )


    # --------------------------------------------------------
    # CATEGORICAL DATA
    # --------------------------------------------------------

    for feature in categorical_features:

        value = input_df[feature].iloc[0]

        if pd.isna(value):

            input_df[feature] = "Unknown"

        else:

            input_df[feature] = (
                input_df[feature]
                .astype(str)
            )


    # --------------------------------------------------------
    # ENCODER
    # --------------------------------------------------------

    try:

        encoded = encoder.transform(
            input_df
        )

        encoded_df = convert_to_dataframe(
            encoded
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=f"Encoding failed: {error}",
        )


    # --------------------------------------------------------
    # IALP
    # --------------------------------------------------------

    try:

        ialp_output = ialp_model.transform(
            encoded_df
        )

        ialp_df = convert_to_dataframe(
            ialp_output
        )

        ialp_array = convert_to_numpy(
            ialp_df
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"IALP processing failed: {error}",
        )


    # --------------------------------------------------------
    # IFF
    # --------------------------------------------------------

    try:

        iff_prediction_array = np.asarray(
            iff_model.predict(
                ialp_array
            )
        ).reshape(-1)

        iff_prediction = int(
            iff_prediction_array[0]
        )

        if hasattr(
            iff_model,
            "score_samples"
        ):

            iff_score = float(
                np.asarray(
                    iff_model.score_samples(
                        ialp_array
                    )
                )
                .reshape(-1)[0]
            )

        else:

            iff_score = 0.0

        iff_anomaly = (
            iff_prediction == -1
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"IFF processing failed: {error}",
        )


    # --------------------------------------------------------
    # FINAL FEATURES
    # --------------------------------------------------------

    try:

        iff_column = np.asarray(
            [iff_prediction],
            dtype=float
        ).reshape(-1, 1)

        score_column = np.asarray(
            [iff_score],
            dtype=float
        ).reshape(-1, 1)

        final_features = np.hstack(
            [
                ialp_array,
                iff_column,
                score_column,
            ]
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Feature construction failed: {error}",
        )


    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    try:

        prediction = "Unknown"
        confidence = 0.0

        if hasattr(
            xgb_model,
            "predict_with_confidence"
        ):

            results = (
                xgb_model
                .predict_with_confidence(
                    final_features
                )
            )

            if results:

                result = results[0]

                prediction = str(
                    result.get(
                        "label",
                        result.get(
                            "prediction",
                            "Unknown"
                        )
                    )
                )

                confidence = float(
                    result.get(
                        "confidence",
                        0.0
                    )
                )

        else:

            raw_prediction = (
                xgb_model.predict(
                    final_features
                )
            )

            if len(raw_prediction) > 0:

                prediction = str(
                    raw_prediction[0]
                )

            if hasattr(
                xgb_model,
                "predict_proba"
            ):

                probabilities = (
                    xgb_model.predict_proba(
                        final_features
                    )
                )

                confidence = (
                    float(
                        np.max(
                            probabilities[0]
                        )
                    )
                    * 100
                )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"XGBoost prediction failed: {error}",
        )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if 0 <= confidence <= 1:
        confidence *= 100

    confidence = max(
        0.0,
        min(
            100.0,
            confidence
        )
    )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    prediction_lower = (
        prediction
        .strip()
        .lower()
    )

    if prediction_lower in [
        "normal",
        "benign",
        "normal traffic",
    ]:

        prediction_display = "NORMAL"

        detection_status = (
            "Normal Traffic"
        )

    else:

        prediction_display = "ATTACK"

        detection_status = (
            "Malicious Traffic"
        )


    return {

        "success":
            True,

        "prediction":
            prediction_display,

        "attack_type":
            prediction,

        "confidence":
            round(
                confidence,
                2
            ),

        "detection_status":
            detection_status,

        "iff_anomaly":
            bool(
                iff_anomaly
            ),

        "iff_score":
            round(
                iff_score,
                6
            ),

        "dataset_type":
            "KDDCup99-compatible",
    }


# ============================================================
# DATASET INFO
# ============================================================

@app.get("/dataset-info")
def dataset_info():

    csv_files = list(
        DATASET_DIR.glob("*.csv")
    )

    if not csv_files:

        raise HTTPException(
            status_code=404,
            detail="No CSV dataset found.",
        )

    dataset_file = csv_files[0]

    try:

        dataframe = pd.read_csv(
            dataset_file
        )

        information = get_dataset_info(
            dataframe
        )

        label_column = information.get(
            "label_column"
        )

        class_distribution = {}

        if (
            label_column
            and
            label_column in dataframe.columns
        ):

            counts = (
                dataframe[label_column]
                .astype(str)
                .value_counts()
                .to_dict()
            )

            class_distribution = {
                str(key): int(value)
                for key, value in counts.items()
            }

        return {

            "success":
                True,

            "filename":
                dataset_file.name,

            "dataset":
                information,

            "compatible_with_current_model":
                bool(
                    is_kdd99_compatible(
                        dataframe
                    )
                ),

            "class_distribution":
                class_distribution,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Could not load dataset information: {error}",
        )


# ============================================================
# MODEL ANALYSIS
# ============================================================

@app.get("/analyze")
def analyze():

    if run_full_analysis is None:

        return {
            "success": False,
            "message":
                "Complete analysis module is not available.",
            "comparison": [],
        }

    try:

        return run_full_analysis()

    except Exception as error:

        print(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=f"Model analysis failed: {error}",
        )


# ============================================================
# UPLOADED DATASET ANALYSIS
# ============================================================

@app.post("/analyze-upload")
async def analyze_uploaded_dataset(
    file: UploadFile = File(...)
):

    try:

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="No file was selected.",
            )

        if not file.filename.lower().endswith(".csv"):

            raise HTTPException(
                status_code=400,
                detail="Please upload a CSV file.",
            )

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        dataframe = pd.read_csv(
            BytesIO(contents)
        )

        if dataframe.empty:

            raise HTTPException(
                status_code=400,
                detail="CSV contains no records.",
            )


        print()
        print("=" * 60)
        print("UPLOADED DATASET ANALYSIS")
        print("=" * 60)
        print("File:", file.filename)
        print("Rows:", len(dataframe))
        print("Columns:", len(dataframe.columns))
        print("=" * 60)


        information = get_dataset_info(
            dataframe
        )

        compatible = is_kdd99_compatible(
            dataframe
        )

        label_column = information.get(
            "label_column"
        )

        class_distribution = {}

        if (
            label_column
            and
            label_column in dataframe.columns
        ):

            counts = (
                dataframe[label_column]
                .astype(str)
                .value_counts()
                .to_dict()
            )

            class_distribution = {
                str(key): int(value)
                for key, value in counts.items()
            }


        # ----------------------------------------------------
        # BENCHMARK
        # ----------------------------------------------------

        benchmark = run_dataset_benchmark(
            dataframe
        )


        comparison = benchmark.get(
            "comparison",
            []
        )


        metrics = {}

        if comparison:

            proposed = None

            for item in comparison:

                name = str(
                    item.get(
                        "model",
                        item.get(
                            "name",
                            ""
                        )
                    )
                ).lower()

                if (
                    "ialp" in name
                    or
                    "proposed" in name
                ):

                    proposed = item
                    break

            if proposed is None:
                proposed = comparison[-1]

            metrics = {
                "accuracy":
                    proposed.get(
                        "accuracy",
                        0
                    ),

                "precision":
                    proposed.get(
                        "precision",
                        proposed.get(
                            "macro_precision",
                            0
                        )
                    ),

                "recall":
                    proposed.get(
                        "recall",
                        proposed.get(
                            "macro_recall",
                            0
                        )
                    ),

                "f1":
                    proposed.get(
                        "f1",
                        proposed.get(
                            "macro_f1",
                            0
                        )
                    ),

                "macro_precision":
                    proposed.get(
                        "macro_precision",
                        proposed.get(
                            "precision",
                            0
                        )
                    ),

                "macro_recall":
                    proposed.get(
                        "macro_recall",
                        proposed.get(
                            "recall",
                            0
                        )
                    ),

                "macro_f1":
                    proposed.get(
                        "macro_f1",
                        proposed.get(
                            "f1",
                            0
                        )
                    ),

                "weighted_f1":
                    proposed.get(
                        "weighted_f1",
                        0
                    ),
            }


        return {

            "success":
                True,

            "filename":
                file.filename,

            "dataset": {

                "filename":
                    file.filename,

                "rows":
                    len(dataframe),

                "records":
                    len(dataframe),

                "columns":
                    len(dataframe.columns),

                "features":
                    (
                        len(dataframe.columns)
                        -
                        (
                            1
                            if label_column
                            else 0
                        )
                    ),

                "feature_count":
                    (
                        len(dataframe.columns)
                        -
                        (
                            1
                            if label_column
                            else 0
                        )
                    ),

                "classes":
                    len(
                        class_distribution
                    ),

                "class_count":
                    len(
                        class_distribution
                    ),

                "class_names":
                    list(
                        class_distribution.keys()
                    ),
            },

            "compatible_with_current_model":
                bool(compatible),

            "label_column":
                label_column,

            "class_distribution":
                class_distribution,

            "metrics":
                metrics,

            "comparison":
                comparison,

            "model_comparison":
                comparison,

            "benchmark":
                benchmark,

            "confusion_matrix":
                benchmark.get(
                    "confusion_matrix",
                    {}
                ),

            "roc":
                benchmark.get(
                    "roc",
                    []
                ),

            "message":
                "Dataset evaluation completed successfully.",
        }


    except HTTPException:
        raise

    except Exception as error:

        print()
        print("UPLOADED DATASET ANALYSIS ERROR")
        print(error)
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Uploaded dataset analysis failed.",

                "error":
                    str(error),
            },
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )