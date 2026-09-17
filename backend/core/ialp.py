import numpy as np
import pandas as pd


class IALP:

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.adaptive_threshold_ = None
        self.n_features_in_ = None

    # ========================================================
    # FIT
    # ========================================================

    def fit(self, X):

        X = self._to_numpy(X)

        if X.ndim != 2:
            raise ValueError(
                "IALP input must be a 2-dimensional dataset."
            )

        self.n_features_in_ = X.shape[1]

        # Calculate mean
        self.mean_ = np.mean(X, axis=0)

        # Calculate standard deviation
        self.std_ = np.std(X, axis=0)

        # Avoid division by zero
        self.std_ = np.where(
            self.std_ < 1e-8,
            1.0,
            self.std_
        )

        # Calculate z-scores
        z_scores = np.abs(
            (X - self.mean_) / self.std_
        )

        # Calculate record-level anomaly score
        feature_scores = np.mean(
            z_scores,
            axis=1
        )

        # Adaptive threshold
        self.adaptive_threshold_ = float(
            np.percentile(
                feature_scores,
                95
            )
        )

        # Safety check
        if not np.isfinite(
            self.adaptive_threshold_
        ):
            self.adaptive_threshold_ = 1.0

        return self

    # ========================================================
    # TRANSFORM
    # ========================================================

    def transform(self, X):

        if self.mean_ is None:
            raise ValueError(
                "IALP model has not been fitted yet."
            )

        X = self._to_numpy(X)

        if X.ndim != 2:
            raise ValueError(
                "IALP input must be a 2-dimensional dataset."
            )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but received {X.shape[1]} features."
            )

        # Calculate z-scores
        z_scores = (
            X - self.mean_
        ) / self.std_

        # Calculate adaptive anomaly score
        adaptive_score = np.mean(
            np.abs(z_scores),
            axis=1,
            keepdims=True
        )

        # Calculate anomaly indicator
        anomaly_indicator = (
            adaptive_score
            > self.adaptive_threshold_
        ).astype(float)

        # Add IALP features
        return np.column_stack(
            [
                X,
                adaptive_score,
                anomaly_indicator
            ]
        )

    # ========================================================
    # FIT TRANSFORM
    # ========================================================

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)

    # ========================================================
    # CONVERT INPUT TO NUMPY
    # ========================================================

    @staticmethod
    def _to_numpy(X):

        if isinstance(X, pd.DataFrame):

            X = X.to_numpy(
                dtype=float
            )

        elif isinstance(X, pd.Series):

            X = X.to_numpy(
                dtype=float
            )

        else:

            X = np.asarray(
                X,
                dtype=float
            )

        # Convert one-dimensional input
        # into one record
        if X.ndim == 1:

            X = X.reshape(
                1,
                -1
            )

        # Replace invalid values
        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return X

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_info(self):

        return {
            "input_features": self.n_features_in_,
            "output_features": (
                self.n_features_in_ + 2
                if self.n_features_in_
                else None
            ),
            "adaptive_threshold": (
                self.adaptive_threshold_
            )
        }