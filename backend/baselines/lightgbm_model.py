from lightgbm import LGBMClassifier


def create_lightgbm():

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=-1,
        random_state=42,
        verbosity=-1
    )

    return model