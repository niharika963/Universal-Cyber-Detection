from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# IMPORTS
# ============================================================

from backend.preprocessing.dataset_adapter import (
    KDD99_FEATURES,
    get_dataset_info,
    load_and_detect_dataset,
    is_kdd99_compatible,
)

from backend.model_analysis import run_full_analysis


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "backend" / "models"
DATASET_DIR = BASE_DIR / "datasets"

ENCODER_PATH = MODEL_DIR / "encoder.pkl"
IALP_PATH = MODEL_DIR / "ialp.pkl"
IFF_PATH = MODEL_DIR / "iff.pkl"
XGBOOST_PATH = MODEL_DIR / "xgboost.pkl"

DEFAULT_DATASET_PATH = DATASET_DIR / "train.csv"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Universal Cyber Attack Detection System",
    description=(
        "Cyber attack detection using Dataset Adapter, "
        "IALP, IFF and XGBoost."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL MODELS
# ============================================================

encoder = None
ialp_model = None
iff_model = None
xgb_model = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def convert_to_dataframe(data):
    """
    Convert numpy arrays / pandas objects into a DataFrame.
    """

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, pd.Series):
        return data.to_frame()

    return pd.DataFrame(data)


def convert_to_numpy(data):
    """
    Convert pandas/numpy data into a numpy array.
    """

    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=float)

    if isinstance(data, pd.Series):
        return data.to_numpy(dtype=float)

    return np.asarray(data, dtype=float)


def load_models():
    """
    Load all trained models.
    """

    global encoder
    global ialp_model
    global iff_model
    global xgb_model

    required_files = [
        ENCODER_PATH,
        IALP_PATH,
        IFF_PATH,
        XGBOOST_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required model files are missing:\n"
            + "\n".join(missing_files)
        )

    encoder = joblib.load(ENCODER_PATH)
    ialp_model = joblib.load(IALP_PATH)
    iff_model = joblib.load(IFF_PATH)
    xgb_model = joblib.load(XGBOOST_PATH)

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


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    try:
        load_models()
    except Exception as error:
        print()
        print("=" * 60)
        print("MODEL LOADING ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)
        print()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Universal Cyber Attack Detection System is running.",
        "status": "online",
        "pipeline": [
            "Dataset Adapter",
            "Cleaning",
            "Encoding",
            "IALP",
            "IFF",
            "XGBoost",
        ],
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ialp_loaded": ialp_model is not None,
        "iff_loaded": iff_model is not None,
        "xgboost_loaded": xgb_model is not None,
        "encoder_loaded": encoder is not None,
    }


# ============================================================
# INSPECT UPLOADED DATASET
# ============================================================

@app.post("/inspect-dataset")
async def inspect_dataset(file: UploadFile = File(...)):
    """
    Inspect a cybersecurity CSV dataset.

    This endpoint DOES NOT train the model.
    It only checks the uploaded dataset.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename was provided."
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty."
            )

        dataframe = pd.read_csv(BytesIO(file_bytes))

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file: {error}"
        )

    try:
        information = get_dataset_info(dataframe)

        compatible = is_kdd99_compatible(dataframe)

        label_column = information.get("label_column")

        class_distribution = {}

        if label_column and label_column in dataframe.columns:
            distribution = (
                dataframe[label_column]
                .astype(str)
                .value_counts()
                .to_dict()
            )

            class_distribution = {
                str(key): int(value)
                for key, value in distribution.items()
            }

        return {
            "filename": file.filename,
            "dataset": information,
            "compatible_with_current_model": bool(compatible),
            "label_column": label_column,
            "feature_count": len(dataframe.columns) - (
                1 if label_column else 0
            ),
            "row_count": len(dataframe),
            "columns": dataframe.columns.tolist(),
            "class_distribution": class_distribution,
            "message": (
                "Dataset is compatible with the current "
                "KDDCup99-trained model."
                if compatible
                else
                "Dataset was detected, but it is not compatible "
                "with the currently trained model."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset inspection failed: {error}"
        )


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(record: dict):
    """
    Predict whether a network record is Normal or an Attack.

    Current trained model:
    KDDCup99-compatible 41-feature input.
    """

    if encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Encoder model is not loaded."
        )

    if ialp_model is None:
        raise HTTPException(
            status_code=503,
            detail="IALP model is not loaded."
        )

    if iff_model is None:
        raise HTTPException(
            status_code=503,
            detail="IFF model is not loaded."
        )

    if xgb_model is None:
        raise HTTPException(
            status_code=503,
            detail="XGBoost model is not loaded."
        )

    if not isinstance(record, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object."
        )

    # --------------------------------------------------------
    # CHECK REQUIRED FEATURES
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in KDD99_FEATURES
        if feature not in record
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Required features are missing.",
                "missing_features": missing_features,
                "required_feature_count": len(KDD99_FEATURES),
                "received_feature_count": len(record),
            },
        )

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    try:
        input_data = {
            feature: record[feature]
            for feature in KDD99_FEATURES
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=KDD99_FEATURES,
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not create input record: {error}"
        )

    # --------------------------------------------------------
    # VALIDATE NUMERICAL FEATURES
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

    invalid_numeric_features = []

    for feature in numeric_features:
        value = pd.to_numeric(
            input_df.at[0, feature],
            errors="coerce",
        )

        if pd.isna(value) or not np.isfinite(float(value)):
            invalid_numeric_features.append(feature)
        else:
            input_df.at[0, feature] = float(value)

    if invalid_numeric_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid numerical values found.",
                "invalid_features": invalid_numeric_features,
            },
        )

    # --------------------------------------------------------
    # CLEAN CATEGORICAL VALUES
    # --------------------------------------------------------

    for feature in categorical_features:
        value = input_df.at[0, feature]

        if pd.isna(value):
            input_df.at[0, feature] = "Unknown"
        else:
            input_df.at[0, feature] = str(value)

    print()
    print("Prediction request received")
    print("Input shape:", input_df.shape)

    # --------------------------------------------------------
    # ENCODING
    # --------------------------------------------------------

    try:
        encoded = encoder.transform(input_df)
        encoded_df = convert_to_dataframe(encoded)

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Encoding failed: {error}"
        )

    print("Encoded shape:", encoded_df.shape)

    # --------------------------------------------------------
    # IALP
    # --------------------------------------------------------

    try:
        ialp_output = ialp_model.transform(encoded_df)
        ialp_df = convert_to_dataframe(ialp_output)
        ialp_array = convert_to_numpy(ialp_df)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"IALP processing failed: {error}"
        )

    print("IALP output shape:", ialp_df.shape)

    # --------------------------------------------------------
    # IFF / ISOLATION FOREST
    # --------------------------------------------------------

    try:
        iff_prediction = int(
            iff_model.predict(ialp_array)[0]
        )

        iff_score = float(
            iff_model.score_samples(ialp_array)[0]
        )

        iff_anomaly = iff_prediction == -1

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"IFF processing failed: {error}"
        )

    # --------------------------------------------------------
    # FINAL FEATURE REPRESENTATION
    # --------------------------------------------------------

    try:
        final_features = np.column_stack(
            [
                ialp_array,
                [iff_prediction],
                [iff_score],
            ]
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Feature construction failed: {error}"
        )

    print(
        "Final XGBoost shape:",
        final_features.shape
    )

    # --------------------------------------------------------
    # XGBOOST PREDICTION
    # --------------------------------------------------------

    try:
        results = xgb_model.predict_with_confidence(
            final_features
        )

        if not results:
            raise ValueError(
                "XGBoost returned no prediction."
            )

        result = results[0]

        prediction = str(result["label"])
        confidence = float(result["confidence"])

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"XGBoost prediction failed: {error}"
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if prediction.lower() == "normal":
        detection_status = "Benign Traffic"
        prediction_display = "NORMAL"
    else:
        detection_status = "Malicious Traffic"
        prediction_display = "ATTACK"

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    response = {
        "prediction": prediction_display,
        "attack_type": prediction,
        "confidence": confidence,
        "detection_status": detection_status,
        "iff_anomaly": bool(iff_anomaly),
        "iff_score": iff_score,
        "dataset_type": "KDDCup99-compatible",
    }

    print()
    print("Prediction:", prediction_display)
    print("Attack Type:", prediction)
    print("Confidence:", confidence)
    print("IFF Anomaly:", iff_anomaly)
    print()

    return response


# ============================================================
# DATASET INFORMATION
# ============================================================

@app.get("/dataset-info")
def dataset_info():
    """
    Return information about the currently available
    training dataset.
    """

    if not DEFAULT_DATASET_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset not found: "
                f"{DEFAULT_DATASET_PATH}"
            ),
        )

    try:
        dataframe = pd.read_csv(
            DEFAULT_DATASET_PATH
        )

        information = get_dataset_info(dataframe)

        label_column = information.get("label_column")

        class_distribution = {}

        if label_column and label_column in dataframe.columns:
            distribution = (
                dataframe[label_column]
                .astype(str)
                .value_counts()
                .to_dict()
            )

            class_distribution = {
                str(key): int(value)
                for key, value in distribution.items()
            }

        return {
            "filename": DEFAULT_DATASET_PATH.name,
            "dataset_path": str(DEFAULT_DATASET_PATH),
            "dataset": information,
            "compatible_with_current_model": bool(
                is_kdd99_compatible(dataframe)
            ),
            "class_distribution": class_distribution,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load dataset information: {error}"
        )


# ============================================================
# FULL MODEL ANALYSIS
# ============================================================

@app.get("/analyze")
def analyze():
    """
    Return the current model analysis.
    """

    try:
        result = run_full_analysis()
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model analysis failed: {error}"
        )


# ============================================================
# REFRESH MODEL ANALYSIS
# ============================================================

@app.post("/analyze/refresh")
def refresh_analysis():
    """
    Re-run model analysis.
    """

    try:
        result = run_full_analysis()
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis refresh failed: {error}"
        )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )