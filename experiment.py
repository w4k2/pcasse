from PCASSE import PCASSE
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
print("LOADED")
# Configure
clfs = {
    'PCASSE': PCASSE(subspace_size=4, n_components=2),
    'SVC': SVC()
}
n_splits = 5
repetitions = 1

# Gather dataset
for n_features in range(100,10000,100):
    scores = np.zeros((len(clfs),n_splits, repetitions))
    for repetition, random_state in enumerate(range(repetitions)):
        X, y = make_classification(n_samples=250, n_features=n_features, random_state=42+random_state)


        # Perform CV
        skf = StratifiedKFold(n_splits=n_splits)
        for split, (train, test) in enumerate(skf.split(X,y)):
            # Iterate classifiers
            for clf_idx, clf_n in enumerate(clfs):
                clf = clone(clfs[clf_n]).fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                score = accuracy_score(y[test], y_pred)
                scores[clf_idx,split,repetition] = score

    mean_repeated_scores = np.mean(scores, axis=2)
    mean_scores = np.mean(mean_repeated_scores, axis=1)

    print("%3i" % n_features, mean_scores, "%.3f" % (mean_scores[0]-mean_scores[1]))
