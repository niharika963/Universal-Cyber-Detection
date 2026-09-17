import pandas as pd


# ============================================================
# KDD CUP 99 FEATURES
# ============================================================

KDD99_FEATURES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


# ============================================================
# KDD99 ATTACK MAPPING
# ============================================================

KDD99_ATTACK_MAPPING = {

    # Normal
    "normal": "Normal",

    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",

    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",

    # R2L
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",

    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
}


# ============================================================
# POSSIBLE LABEL COLUMN NAMES
# ============================================================

LABEL_COLUMN_CANDIDATES = [
    "label",
    "class",
    "attack",
    "attack_type",
    "attacktype",
    "connection_type",
    "connectiontype",
    "target",
    "category",
]


# ============================================================
# COLUMN NAME NORMALIZATION
# ============================================================

def normalize_column_name(column):
    """
    Convert a column name into a standard format.
    """

    column = str(column).strip().lower()

    column = column.replace(" ", "_")
    column = column.replace("-", "_")
    column = column.replace("/", "_")

    return column


def normalize_dataframe_columns(df):
    """
    Normalize all dataframe column names.
    """

    df = df.copy()

    df.columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    return df


# ============================================================
# FIND LABEL COLUMN
# ============================================================

def find_label_column(df):
    """
    Find the likely label column in a dataset.
    """

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in LABEL_COLUMN_CANDIDATES:

        candidate = normalize_column_name(candidate)

        if candidate in normalized_columns:
            return normalized_columns[candidate]

    return None


# ============================================================
# DETECT DATASET TYPE
# ============================================================

def detect_dataset_type(df):
    """
    Detect whether the dataset matches KDDCup99.
    """

    normalized_columns = {
        normalize_column_name(column)
        for column in df.columns
    }

    kdd_features = set(KDD99_FEATURES)

    matching_features = (
        normalized_columns.intersection(kdd_features)
    )

    label_column = find_label_column(df)

    if (
        len(matching_features) >= 40
        and label_column is not None
    ):
        return "KDDCup99"

    if len(matching_features) >= 30:
        return "KDDCup99-compatible"

    if label_column is not None:
        return "Generic-Cybersecurity-Dataset"

    return "Unknown-Dataset"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """
    Load a CSV dataset.
    """

    df = pd.read_csv(file_path)

    df = normalize_dataframe_columns(df)

    return df


# ============================================================
# LOAD AND DETECT
# ============================================================

def load_and_detect_dataset(file_path):
    """
    Load dataset and detect its type.
    """

    df = load_dataset(file_path)

    information = get_dataset_info(df)

    return df, information


# ============================================================
# ADAPT KDD99
# ============================================================

def adapt_kdd99(df):
    """
    Convert KDDCup99 dataset into:

        X = features
        y = attack categories

    Categories:

        Normal
        DoS
        Probe
        R2L
        U2R
    """

    df = normalize_dataframe_columns(df.copy())

    label_column = find_label_column(df)

    if label_column is None:
        raise ValueError(
            "Could not find the label column in the KDD99 dataset."
        )

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in KDD99_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "KDD99 dataset is missing required features: "
            + ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # Create feature dataframe
    # --------------------------------------------------------

    X = df[KDD99_FEATURES].copy()

    # --------------------------------------------------------
    # Normalize raw labels
    # --------------------------------------------------------

    raw_labels = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Map attack labels
    # --------------------------------------------------------

    y = raw_labels.map(KDD99_ATTACK_MAPPING)

    # --------------------------------------------------------
    # Check unknown labels
    # --------------------------------------------------------

    unknown_labels = sorted(
        raw_labels[y.isna()].unique().tolist()
    )

    if unknown_labels:

        print()
        print("Warning: Unknown KDD99 labels found:")

        for label in unknown_labels:
            print(" -", label)

        # Treat unknown labels as Normal only if they are
        # genuinely not attack labels is NOT safe.
        # Therefore raise an error instead.
        raise ValueError(
            "Unknown KDD99 attack labels found: "
            + ", ".join(unknown_labels)
        )

    y = y.astype(str)

    print()
    print("KDDCup99 Adapter")
    print("-------------------------")
    print("Features:", X.shape)
    print("Labels:", y.shape)

    print()
    print("Attack Category Distribution:")
    print(y.value_counts())

    return X, y


# ============================================================
# GENERIC DATASET ADAPTER
# ============================================================

def adapt_dataset(df):
    """
    Adapt a supported cybersecurity dataset.

    Currently KDDCup99 is fully supported.
    Other datasets are detected but require their own
    feature/label adapter before model training.
    """

    df = normalize_dataframe_columns(df.copy())

    information = get_dataset_info(df)

    dataset_type = information["dataset_type"]

    if dataset_type in [
        "KDDCup99",
        "KDDCup99-compatible",
    ]:
        return adapt_kdd99(df)

    raise ValueError(
        f"Dataset type '{dataset_type}' is detected, "
        "but a trained adapter/model is not yet available "
        "for this dataset."
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_info(df):
    """
    Return useful information about a dataset.
    """

    df = normalize_dataframe_columns(df.copy())

    label_column = find_label_column(df)

    dataset_type = detect_dataset_type(df)

    feature_count = len(df.columns)

    if label_column is not None:
        feature_count -= 1

    class_distribution = {}

    if label_column is not None:

        distribution = (
            df[label_column]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        class_distribution = {
            str(key): int(value)
            for key, value in distribution.items()
        }

    return {
        "dataset_type": dataset_type,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "feature_count": int(feature_count),
        "label_column": label_column,
        "class_distribution": class_distribution,
        "features": df.columns.tolist(),
    }


# ============================================================
# KDD99 COMPATIBILITY CHECK
# ============================================================

def is_kdd99_compatible(df):
    """
    Check whether a dataframe contains all 41 KDD99
    features and a label column.
    """

    df = normalize_dataframe_columns(df.copy())

    label_column = find_label_column(df)

    if label_column is None:
        return False

    missing_features = [
        feature
        for feature in KDD99_FEATURES
        if feature not in df.columns
    ]

    return len(missing_features) == 0


# ============================================================
# SUPPORTED DATASETS
# ============================================================

def get_supported_datasets():
    """
    Return currently supported datasets.
    """

    return {
        "KDDCup99": {
            "status": "supported",
            "features": 41,
            "attack_categories": [
                "Normal",
                "DoS",
                "Probe",
                "R2L",
                "U2R",
            ],
        },

        "Generic-Cybersecurity-Dataset": {
            "status": "detection-only",
            "message": (
                "Dataset can be inspected, but a "
                "dataset-specific adapter and trained "
                "model are required before prediction."
            ),
        },
    }