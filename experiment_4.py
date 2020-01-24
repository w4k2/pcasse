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
n_components = 20
n_samples = 150
features = [100, 1000, 10000]

dts = np.linspace(.01,.99,10)
print(dts)

#exit()

for n_features in features:
    print("%i features" % n_features)
    scores = np.ones((
        repetitions, len(dts), n_splits, 2
    ))

    for repetition in range(repetitions):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_components,
            n_clusters_per_class=2,
            n_redundant=0,
            random_state=1410 + repetition,
        )

        for dt_idx, dt in enumerate(dts):
            #print(dt)
            clfs = {
                "SVC": SVC(),
                "PCASSE": PCASSEE(distribuant_treshold=dt),
            }
            #print(clfs.keys())

            skf = StratifiedKFold(n_splits=n_splits)
            for split, (train, test) in enumerate(skf.split(X, y)):
                # Iterate classifiers
                for clf_idx, clf_n in enumerate(clfs):
                    clf = clone(clfs[clf_n]).fit(X[train], y[train])
                    y_pred = clf.predict(X[test])
                    score = accuracy_score(y[test], y_pred)
                    scores[repetition, dt_idx, split, clf_idx] = score

                    print("n_f %i r %i dt %i s %i c %i"% (n_features, repetition, dt_idx, split, clf_idx))

                    #print(clf_n, "%.3f" % score)
                    #scores[clf_idx, split, repetition] = score

    stabilized_scores = np.mean(scores, axis=(0,2))
    np.save("results-dts-%if" % n_features, stabilized_scores)

    print(stabilized_scores, stabilized_scores.shape)
exit()

"""
    exit()
    # Configure

    overall_scores = []

    # Gather dataset
    for n_features in range(n_components, 220, 20):
        print("%i features" % n_features)
        scores = np.zeros((len(clfs), n_splits, repetitions))
        for repetition, random_state in enumerate(range(repetitions)):

            # Perform CV

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
    np.save("results_%i-0" % n_samples, overall_scores)
"""
