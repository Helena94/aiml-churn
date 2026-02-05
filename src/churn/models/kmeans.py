from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from churn.features.preprocessing import build_preprocessor


def _silhouette_scorer(estimator, X):
    """Custom scorer for KMeans using silhouette score."""
    labels = estimator.predict(X)
    if len(set(labels)) < 2:
        return -1
    return silhouette_score(X, labels)


def build_kmeans_model(params_grid=None, SearchCVConfig=None):
    """
    Returns a GridSearchCV object
    for KMeans clustering.
    """
    pipe = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", KMeans(init="k-means++", n_init=10, random_state=42)),
        ]
    )

    param_grid = params_grid or {
        "model__n_clusters": [2, 3, 4, 5, 6, 7, 8],
    }

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=_silhouette_scorer,
        cv=SearchCVConfig.get("cv", 5) if SearchCVConfig else 5,
        n_jobs=SearchCVConfig.get("n_jobs", -1) if SearchCVConfig else -1,
        verbose=SearchCVConfig.get("verbose", 1) if SearchCVConfig else 1,
    )

    return search
