"""
- Ustalona liczba cech informatywnych.
- Różna liczba wzorców.
- Wzrastająca liczba cech ogółem.
"""
from PCASSE import PCASSE, PCASSEE, RS
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


n_splits = 5
repetitions = 10
max = 30000
n_components = 20

for n_samples in [100, 150, 200]:
    print("LOADED %i" % n_samples)
    # Configure
    clfs = {
        "SVC": SVC(),
        "RS": RS(),
        "PCASSE-Oracle": PCASSE(n_components=n_components),
        "PCASSE.1": PCASSEE(distribuant_treshold=0.1),
        "PCASSE.2": PCASSEE(distribuant_treshold=0.2),
        "PCASSE.3": PCASSEE(distribuant_treshold=0.3),
    }

    print(clfs.keys())
    overall_scores = []

    # Gather dataset
    for n_features in range(1000, max, 1000):
        scores = np.zeros((len(clfs), n_splits, repetitions))
        for repetition, random_state in enumerate(range(repetitions)):
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_components,
                n_clusters_per_class=2,
                n_redundant=0,
                random_state=1410 + random_state,
            )

            # Perform CV
            skf = StratifiedKFold(n_splits=n_splits)
            for split, (train, test) in enumerate(skf.split(X, y)):
                # Iterate classifiers
                for clf_idx, clf_n in enumerate(clfs):
                    clf = clone(clfs[clf_n]).fit(X[train], y[train])
                    y_pred = clf.predict(X[test])
                    score = accuracy_score(y[test], y_pred)
                    scores[clf_idx, split, repetition] = score

        mean_repeated_scores = np.mean(scores, axis=2)
        mean_scores = np.mean(mean_repeated_scores, axis=1)

        print(
            "%5i" % n_features,
            mean_scores,
            # "%.3f" % (mean_scores[0] - mean_scores[1]),
            # clf.n_components,
        )
        overall_scores.append(mean_scores)

    overall_scores = np.array(overall_scores)
    print(overall_scores)
    np.save("results_%i-2" % n_samples, overall_scores)
