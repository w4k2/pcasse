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
        components = np.abs(
            PCA(n_components=self.n_components, svd_solver="full").fit(X).components_
        )
        # print("A ", components.shape)

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


class PCASSEE(ClassifierMixin, BaseEstimator):
    def __init__(self, subspace_size=None, distribuant_treshold=0.1):
        self.subspace_size = subspace_size
        self.distribuant_treshold = distribuant_treshold

    def fit(self, X, y, classes=None):
        # Calculate PCA components

        pca = PCA(svd_solver="full").fit(X)
        components = np.abs(pca.components_)

        # Z EVR
        evrd = np.add.accumulate(pca.explained_variance_ratio_)
        self.n_components = np.where(evrd > self.distribuant_treshold)[0][0]
        components = components[: self.n_components, :]

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
