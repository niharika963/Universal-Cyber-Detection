import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class DataEncoder:

    def __init__(self):

        self.encoder = None

        self.categorical_columns = []

        self.numeric_columns = []


    # ========================================================
    # FIT + TRANSFORM
    # ========================================================

    def fit_transform(self, X):

        X = X.copy()

        self.categorical_columns = (
            X.select_dtypes(
                include=[
                    "object",
                    "string",
                    "category"
                ]
            )
            .columns
            .tolist()
        )

        self.numeric_columns = (
            X.select_dtypes(
                exclude=[
                    "object",
                    "string",
                    "category"
                ]
            )
            .columns
            .tolist()
        )

        print("Categorical columns:")
        print(self.categorical_columns)

        print("\nNumerical columns:")
        print(self.numeric_columns)

        self.encoder = ColumnTransformer(

            transformers=[

                (
                    "categorical",

                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),

                    self.categorical_columns
                ),

                (
                    "numeric",

                    "passthrough",

                    self.numeric_columns
                )
            ],

            remainder="drop"
        )

        encoded = self.encoder.fit_transform(X)

        return pd.DataFrame(encoded)


    # ========================================================
    # TRANSFORM
    # ========================================================

    def transform(self, X):

        if self.encoder is None:

            raise ValueError(
                "Encoder has not been fitted yet."
            )

        X = X.copy()

        encoded = self.encoder.transform(X)

        return pd.DataFrame(encoded)