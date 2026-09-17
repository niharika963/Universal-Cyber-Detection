# ============================================================
# IFF - ISOLATION FOREST
# Universal Cyber Attack Detection System
# ============================================================

import numpy as np

from sklearn.ensemble import IsolationForest


class IFF:

    def __init__(
        self,
        n_estimators=150,
        contamination="auto",
        random_state=42
    ):

        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        self.model = None

    # ========================================================
    # FIT
    # ========================================================

    def fit(self, X):

        X = self._to_numpy(X)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(X)

        return self

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, X):

        self._check_fitted()

        X = self._to_numpy(X)

        return self.model.predict(X)

    # ========================================================
    # SCORE
    # ========================================================

    def score_samples(self, X):

        self._check_fitted()

        X = self._to_numpy(X)

        return self.model.score_samples(X)

    # ========================================================
    # DECISION FUNCTION
    # ========================================================

    def decision_function(self, X):

        self._check_fitted()

        X = self._to_numpy(X)

        return self.model.decision_function(X)

    # ========================================================
    # CHECK
    # ========================================================

    def _check_fitted(self):

        if self.model is None:

            raise ValueError(
                "IFF model has not been fitted."
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