#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

from OPLS import PLS, OPLS, OPLS_PLS


with np.load(pathlib.Path(__file__).parent.parent / "rsc" / "test_data.npz") as npz:
    X = npz["X"]
    y = npz["y"]
    Y = npz["Y"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

if (True):
    fig, axes = plt.subplots(1, 2)

    for center, scale in ((True, True), (True, False), (False, False)):
        print(f"center={center}, scale={scale}")
        ncomp = []
        opls = []
        pls = []
        for i in range(1, 16):
            ncomp.append(i)
            opls.append(OPLS_PLS(n_components=1, pls_components=i, center=center,
                        scale=scale).fit(X_train, y_train).score(X_test, y_test))
            pls.append(PLS(n_components=i, center=center, scale=scale
                           ).fit(X_train, y_train).score(X_test, y_test))
        axes[0].plot(ncomp, pls,  label=f"center={center}, scale={scale}")
        axes[1].plot(ncomp, opls, label=f"center={center}, scale={scale}")

        axes[0].set_title("Pure PLS")
        axes[1].set_title("OPLS filtering with one component")

        for ax in axes:
            ax.legend()
            ax.set_xlabel("number of components")
            ax.set_ylabel("Test score")

    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig(pathlib.Path(__file__).parent / "center_scale_test.png")

    quit()

if (False):
    maxk = 15

    ncomp = np.arange(maxk)+1
    pls_score = []
    for k in ncomp:
        print(k, end="\r")
        pls = PLS(n_components=k).fit(X_train, y_train)
        pls_score.append(pls.score(X_test, y_test))

    print("pls ", ["%.4f" % v for v in pls_score])
    opls_score = []
    for k in ncomp:
        print(k, end="\r")
        pls = OPLS(n_components=k).fit(X_train, y_train)
        opls_score.append(pls.score(X_test, y_test))

    print("opls", ["%.4f" % v for v in opls_score])

    opls_pls_score = {}
    ncomp_opls = np.arange(1, 7)
    for i in ncomp_opls:
        opls_pls_score[i] = []
        for k in ncomp:
            print(i, k, end="\r")
            opls = OPLS(n_components=i).fit(X_train, y_train)
            pls = PLS(n_components=k).fit(opls.transform(X_train), y_train)
            opls_pls_score[i].append(pls.score(opls.transform(X_test), y_test))

        print("%2d  " % i, ["%.4f" % v for v in opls_pls_score[i]])

    fig, ax = plt.subplots(1)
    ax.plot(ncomp, pls_score, label="PLS")
    ax.plot(ncomp, opls_score, label="OPLS")
    for i in opls_pls_score:
        ax.plot(ncomp, opls_pls_score[i], "--", label=f"OPLS({i})-PLS")

    ax.legend()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(pathlib.Path(__file__).parent / "scores.png")

jaas = 10
print(0, jaas,
      PLS(n_components=10).fit(X_train, y_train).score(X_test, y_test),
      OPLS_PLS(0, 10).fit(X_train, y_train).score(X_test, y_test))
for i in range(1, jaas):
    k = jaas-i
    opls = OPLS(n_components=i).fit(X_train, y_train)
    pls = PLS(n_components=k).fit(opls.transform(X_train), y_train)
    print(i, k, pls.score(opls.transform(X_test), y_test),
          OPLS_PLS(i, k).fit(X_train, y_train).score(X_test, y_test))

print(10, jaas,
      OPLS(n_components=10).fit(X_train, y_train).score(X_test, y_test),
      OPLS_PLS(10, 0).fit(X_train, y_train).score(X_test, y_test))
