import numpy as np
import matplotlib.pyplot as plt
from PCASSE import PCASSE, PCASSEE, RS
from sklearn.svm import SVC
from scipy.signal import medfilt


for n_samples in [100, 150, 200]:
    plt.clf()
    scores = np.load("results_%i.npy" % n_samples)

    clfs = {
        "SVC": SVC(),
        "RS": RS(),
        "PCASSE-Oracle": PCASSE(),
        "PCASSE.1": PCASSEE(distribuant_treshold=0.1),
        "PCASSE.2": PCASSEE(distribuant_treshold=0.2),
        "PCASSE.3": PCASSEE(distribuant_treshold=0.3),
    }

    print(scores)

    for i, clf in enumerate(clfs):
        print(scores[:, i])

        plt.plot(medfilt(scores[:, i], 3), label=clf)
    plt.legend()
    plt.ylim(0, 1)
    plt.title("%i training samples" % n_samples)
    plt.tight_layout()
    plt.savefig("foo")
    plt.savefig("figures/%i.png" % n_samples)
