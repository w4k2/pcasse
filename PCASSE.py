from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

class PCASSE(ClassifierMixin, BaseEstimator):
    def __init__(self, subspace_size=None, n_components=4):
        self.subspace_size = subspace_size
        self.n_components = n_components

    def fit(self, X, y, classes=None):
        # Calculate PCA components
        components = np.abs(PCA(n_components=self.n_components, svd_solver="full").fit(X).components_)

        # Gather ensemble
        self.subspaces = np.array(
            [np.argsort(-row)[: self.subspace_size] for row in components]
        )

        # Build ensemble
        self.ensemble = [SVC().fit(X[:, subspace], y) for subspace in self.subspaces]

        return self

    def predict(self, X):
        return (
            np.mean(
                np.array(
                    [
                        self.ensemble[i].decision_function(X[:, subspace])
                        for i, subspace in enumerate(self.subspaces)
                    ]
                ),
                axis=0,
            )
            > 0
        ).astype(int)
