import pandas as pd
import json
import urllib.request
import os
import sys

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from backend.preprocessing.dataset_adapter import KDD99_FEATURES

CSV_PATH = "datasets/train.csv"
API_URL = "http://127.0.0.1:8000/predict"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Pick one real row from each attack category
rows = {
    "Normal": df[df["connection_type"] == "normal"].iloc[0],
    "DoS": df[df["connection_type"].isin(
        ["back", "land", "neptune", "pod", "smurf", "teardrop"]
    )].iloc[0],
    "Probe": df[df["connection_type"].isin(
        ["ipsweep", "nmap", "portsweep", "satan"]
    )].iloc[0],
    "R2L": df[df["connection_type"].isin(
        ["ftp_write", "guess_passwd", "imap", "multihop",
         "phf", "spy", "warezclient", "warezmaster"]
    )].iloc[0],
    "U2R": df[df["connection_type"].isin(
        ["buffer_overflow", "loadmodule", "perl", "rootkit"]
    )].iloc[0],
}


def test_prediction(category, row):

    # Keep only the 41 KDD99 features
    record = {}

    for feature in KDD99_FEATURES:
        value = row[feature]

        # Convert NumPy values to normal Python values
        if hasattr(value, "item"):
            value = value.item()

        record[feature] = value

    # Convert to JSON
    data = json.dumps(record).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))

        print("\n========================================")
        print("CATEGORY:", category)
        print("Original Label:", row["connection_type"])
        print("----------------------------------------")
        print("Prediction     :", result["prediction"])
        print("Attack Type    :", result["attack_type"])
        print("Confidence     :", result["confidence"], "%")
        print("Status         :", result["detection_status"])
        print("IFF Anomaly     :", result["iff_anomaly"])
        print("IFF Score       :", result["iff_score"])
        print("========================================")

    except Exception as e:
        print("\nERROR:", e)


# Test all categories
for category, row in rows.items():
    test_prediction(category, row)