import pandas as pd
import numpy as np


def clean_data(df):

    df = df.copy()

    # Replace infinity values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Find categorical columns
    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns

    # Fill categorical missing values
    for column in categorical_columns:

        df[column] = df[column].fillna(
            "Unknown"
        )

    # Find numerical columns
    numerical_columns = df.select_dtypes(
        exclude=[
            "object",
            "string",
            "category"
        ]
    ).columns

    # Fill numerical missing values
    for column in numerical_columns:

        if df[column].isna().any():

            df[column] = df[column].fillna(
                df[column].median()
            )

    return df.reset_index(drop=True)