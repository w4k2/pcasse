import numpy as np
import matplotlib.pyplot as plt
from PCASSE import PCASSE, PCASSEE, RS
from sklearn.svm import SVC
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler

methods = ["SVC", "PCASSE"]
colors = [
    (0, 0, 0),
    (0.7, 0, 0),
]
ls = ["-", "-"]
lw = (
    1,
    1,
)

features = [100, 1000, 10000]

dts = np.linspace(0.01, 0.99, 10)
print(dts)

for n_features in features:
    plt.clf()
    plt.figure(figsize=(4.5, 4), dpi=200)

    print("%i features" % n_features)

    scores = np.load("results-dts-%if.npy" % (n_features))
    print(scores)
    for i, method in enumerate(methods):

        x = scores[:, i]
        from scipy.ndimage.filters import gaussian_filter1d

        x_smoothed = gaussian_filter1d(x, sigma=1.5, mode="nearest")
        # x_smoothed=x

        plt.plot(x_smoothed, label=method, c=colors[i], ls=ls[i], lw=lw[i])

    plt.legend(loc=9, ncol=len(methods) // 2, frameon=False)
    plt.ylim(0.45, 1)

    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.99], [".5", ".6", ".7", ".8", ".9", "1."])
    plt.xlim(0, scores.shape[0] - 1)
    plt.grid(ls=":", c=(0.7, 0.7, 0.7))

    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.xticks(
        range(10), ["%.2f" % dt for dt in dts],
    )

    plt.ylabel("Accuracy score")
    plt.xlabel("Explained variance treshold")
    plt.title("%i training features (20 informative)" % n_features)
    plt.tight_layout()

    plt.savefig("foo")
    plt.savefig("figures/%if.png" % (n_features))
