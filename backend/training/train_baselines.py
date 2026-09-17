import time
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    elapsed = time.time() - start

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    print(
        f"Time      : {elapsed:.2f} seconds"
    )

    return {
        "Algorithm": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-Score": f1 * 100
    }


def run_baselines(
    X_train,
    X_test,
    y_train,
    y_test
):

    results = []

    # 1. Decision Tree
    dt = DecisionTreeClassifier(
        random_state=42
    )

    results.append(
        evaluate_model(
            "Decision Tree",
            dt,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )


    # 2. SVM
    svm = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    results.append(
        evaluate_model(
            "SVM",
            svm,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )


    # 3. Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    results.append(
        evaluate_model(
            "Random Forest",
            rf,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )


    # 4. Gradient Boosting
    gbm = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )

    results.append(
        evaluate_model(
            "Gradient Boosting",
            gbm,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )


    return results