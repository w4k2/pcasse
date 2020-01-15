import numpy as np
import matplotlib.pyplot as plt
from PCASSE import PCASSE, PCASSEE
from sklearn.svm import SVC
from scipy.signal import medfilt

scores = np.load("results.npy")

n_components = 20
clfs = {
    "SVC": SVC(),
    "PCASSE-Oracle": PCASSE(subspace_size=4, n_components=n_components),
    "PCASSE.1": PCASSEE(subspace_size=4, distribuant_treshold=0.1),
    "PCASSE.2": PCASSEE(subspace_size=4, distribuant_treshold=0.2),
    "PCASSE.3": PCASSEE(subspace_size=4, distribuant_treshold=0.3),
}

print(scores)

dup = list(range(500, 3500, 500))


for i, clf in enumerate(clfs):
    print(scores[:,i])

    plt.plot(medfilt(scores[:,i],5), label=clf)
plt.legend()
plt.ylim(.45,1)
plt.tight_layout()
plt.savefig("foo")
