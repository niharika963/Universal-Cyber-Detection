# ============================================================
# XGBOOST CYBER ATTACK CLASSIFIER
# ============================================================

import numpy as np

from xgboost import XGBClassifier


class XGBoostModel:

    def __init__(
        self,
        random_state=42
    ):

        self.random_state = random_state

        self.model = None

        self.label_mapping = {
            "Normal": 0,
            "DoS": 1,
            "Probe": 2,
            "R2L": 3,
            "U2R": 4
        }

        self.reverse_mapping = {
            0: "Normal",
            1: "DoS",
            2: "Probe",
            3: "R2L",
            4: "U2R"
        }

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        X,
        y
    ):

        X = self._to_numpy(X)

        y_encoded = np.array(
            [
                self.label_mapping[str(label)]
                for label in y
            ],
            dtype=int
        )

        # ----------------------------------------------------
        # Balanced class weights
        # ----------------------------------------------------

        class_counts = np.bincount(
            y_encoded,
            minlength=5
        )

        total_samples = len(
            y_encoded
        )

        number_of_classes = 5

        class_weights = {}

        for class_id, count in enumerate(
            class_counts
        ):

            if count > 0:

                class_weights[class_id] = (
                    total_samples
                    /
                    (
                        number_of_classes
                        * count
                    )
                )

        sample_weights = np.array(
            [
                class_weights[class_id]
                for class_id in y_encoded
            ]
        )

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        self.model = XGBClassifier(

            objective="multi:softprob",

            num_class=5,

            n_estimators=200,

            max_depth=6,

            learning_rate=0.1,

            subsample=0.8,

            colsample_bytree=0.8,

            eval_metric="mlogloss",

            random_state=self.random_state,

            n_jobs=-1,

            tree_method="hist"
        )

        self.model.fit(
            X,
            y_encoded,
            sample_weight=sample_weights
        )

        return self

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        X
    ):

        self._check_fitted()

        X = self._to_numpy(X)

        predictions = self.model.predict(
            X
        )

        predictions = np.asarray(
            predictions
        ).astype(int)

        return np.array(
            [
                self.reverse_mapping[
                    int(value)
                ]
                for value in predictions
            ]
        )

    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    def predict_proba(
        self,
        X
    ):

        self._check_fitted()

        X = self._to_numpy(X)

        return self.model.predict_proba(
            X
        )

    # ========================================================
    # PREDICT WITH CONFIDENCE
    # ========================================================

    def predict_with_confidence(
        self,
        X
    ):

        probabilities = self.predict_proba(
            X
        )

        predicted_indices = np.argmax(
            probabilities,
            axis=1
        )

        results = []

        for i, index in enumerate(
            predicted_indices
        ):

            label = self.reverse_mapping[
                int(index)
            ]

            confidence = (
                probabilities[i][index]
                * 100
            )

            results.append(
                {
                    "label": label,
                    "confidence": float(
                        confidence
                    )
                }
            )

        return results

    # ========================================================
    # CHECK FIT
    # ========================================================

    def _check_fitted(self):

        if self.model is None:

            raise ValueError(
                "XGBoost model has not been fitted."
            )

    # ========================================================
    # SAFE NUMPY
    # ========================================================

    @staticmethod
    def _to_numpy(X):

        if hasattr(
            X,
            "to_numpy"
        ):

            X = X.to_numpy(
                dtype=float
            )

        else:

            X = np.asarray(
                X,
                dtype=float
            )

        if X.ndim == 1:

            X = X.reshape(
                1,
                -1
            )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return X