from catboost import CatBoostClassifier


def create_catboost():

    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False
    )

    return model