import numpy as np
import matplotlib.pyplot as plt
from PCASSE import PCASSE, PCASSEE, RS
from sklearn.svm import SVC
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler

methods = ["SVC", "RS", "ORACLE", "PCASSE.1", "PCASSE.2", "PCASSE.3"]
colors = [
    (0, 0, 0),
    (0, 0, 0.7),
    (0.7, 0, 0),
    (0.7, 0, 0),
    (0.7, 0, 0),
    (0.7, 0, 0),
]
ls = ["-", "-", ":", "--", "-.", "-"]
lw = 1, 1, 4, 1, 1, 1

for experiment in [0, 1, 2]:
    for n_samples in [100, 150, 200]:
        plt.clf()
        plt.figure(figsize=(4.5, 4), dpi=200)

        scores = np.load("results_%i-%i.npy" % (n_samples, experiment))

        for i, method in enumerate(methods):

            x = scores[:, i]
            from scipy.ndimage.filters import gaussian_filter1d

            x_smoothed = gaussian_filter1d(x, sigma=1.5, mode="nearest")
            # x_smoothed[0] = (x[0] + x_smoothed[0]) / 2
            # x_smoothed[-1] = (x[-1] + x_smoothed[-1]) / 2

            plt.plot(x_smoothed, label=method, c=colors[i], ls=ls[i], lw=lw[i])

        plt.legend(loc=9, ncol=len(methods) // 2, frameon=False)
        plt.ylim(0.45, 1)
        plt.yticks(
            [0.5, 0.6, 0.7, 0.8, 0.9, 0.99], [".5", ".6", ".7", ".8", ".9", "1."]
        )
        print(scores.shape)
        plt.xlim(0, scores.shape[0] - 1)
        plt.grid(ls=":", c=(0.7, 0.7, 0.7))
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if experiment == 2:
            plt.xticks(
                [0, 4, 8, 12, 16, 20, 24, 28],
                [1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000],
            )
        elif experiment == 1:
            plt.xticks(
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                [200, 300, 400, 500, 600, 700, 800, 900, 1000],
            )
        else:
            plt.xticks(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
            )
            print("SMALLEST")

        plt.ylabel("Accuracy score")
        plt.xlabel("Number of features (20 informative)")
        plt.title("%i training samples" % n_samples)
        plt.tight_layout()
        plt.savefig("foo")
        plt.savefig("figures/%i-%i.png" % (n_samples, experiment))
