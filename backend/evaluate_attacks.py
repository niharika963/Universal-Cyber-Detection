import os
import sys
import pandas as pd
import json
import urllib.request
from collections import defaultdict

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from backend.preprocessing.dataset_adapter import ATTACK_CATEGORY_MAP


CSV_PATH = "datasets/train.csv"
API_URL = "http://127.0.0.1:8000/predict"

df = pd.read_csv(CSV_PATH)

# Test one representative row from every original attack subtype
results = []

for attack_name, category in ATTACK_CATEGORY_MAP.items():

    subset = df[
        df["connection_type"].astype(str).str.strip().str.lower()
        == attack_name
    ]

    if subset.empty:
        continue

    # Use the first available row
    row = subset.iloc[0]

    record = {}

    for feature in [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]:
        value = row[feature]

        if hasattr(value, "item"):
            value = value.item()

        record[feature] = value

    try:
        data = json.dumps(record).encode("utf-8")

        request = urllib.request.Request(
            API_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(request) as response:
            prediction = json.loads(
                response.read().decode("utf-8")
            )

        predicted_category = prediction["attack_type"]

        correct = predicted_category == category

        results.append({
            "Original": attack_name,
            "Category": category,
            "Prediction": predicted_category,
            "Correct": correct,
            "Confidence": prediction["confidence"],
            "IFF": prediction["iff_anomaly"]
        })

    except Exception as e:
        print(f"Error testing {attack_name}: {e}")


# Display results
print("\n")
print("=" * 80)
print("        KDDCup99 ATTACK SUBTYPE EVALUATION")
print("=" * 80)

for result in results:

    status = "PASS" if result["Correct"] else "FAIL"

    print(
        f"{result['Original']:18}"
        f"{result['Category']:8}"
        f"{result['Prediction']:10}"
        f"{result['Confidence']:8.2f}%"
        f"  IFF={str(result['IFF']):5}"
        f"  [{status}]"
    )

print("=" * 80)

# Overall result
total = len(results)
correct = sum(r["Correct"] for r in results)

if total > 0:
    accuracy = (correct / total) * 100

    print(f"\nSubtype tests : {total}")
    print(f"Correct       : {correct}")
    print(f"Incorrect     : {total - correct}")
    print(f"Sample accuracy: {accuracy:.2f}%")

# Category summary
print("\n")
print("=" * 80)
print("             CATEGORY SUMMARY")
print("=" * 80)

category_stats = defaultdict(lambda: [0, 0])

for r in results:
    category_stats[r["Category"]][1] += 1

    if r["Correct"]:
        category_stats[r["Category"]][0] += 1

for category, (correct_count, total_count) in category_stats.items():

    percentage = (correct_count / total_count) * 100

    print(
        f"{category:10} : "
        f"{correct_count}/{total_count} "
        f"({percentage:.2f}%)"
    )

print("=" * 80)