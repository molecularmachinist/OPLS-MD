#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

from OPLS_MD import PLS, OPLS, OPLS_PLS, PLS_MD


with np.load(pathlib.Path(__file__).parent.parent / "rsc" / "test_data.npz") as npz:
    X = npz["X"]
    y = npz["y"]
    Y = npz["Y"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

if (True):
    import MDAnalysis as mda
    u = mda.Universe(
        str(pathlib.Path(__file__).parent.parent / "rsc" / "tmd.pdb")
    )
    u.load_new(X_train.reshape((X_train.shape[0], u.atoms.n_atoms, 3)))

    pls_md = PLS_MD(n_components=5).fit(u, y_train)
    pls = PLS(n_components=5).fit(X_train, y_train)
    pls_flip = PLS(n_components=5, flip=True).fit(X_train, y_train)
    pls_sklearn = PLSRegression(n_components=5).fit(X_train, y_train)
    u.load_new(X_test.reshape((X_test.shape[0], u.atoms.n_atoms, 3)))
    assert np.all(np.abs(pls_md.transform(u)-pls.transform(X_test)) < 1e-6)
    assert np.abs(pls_md.score(u, y_test)-pls.score(X_test, y_test)) < 1e-10
    assert np.all(
        np.abs(pls_sklearn.transform(X_test)-pls_flip.transform(X_test)) < 1e-6
    )
    assert np.abs(pls_sklearn.score(X_test, y_test) -
                  pls_flip.score(X_test, y_test)) < 1e-10
    assert np.abs(pls_sklearn.score(X_test, y_test) -
                  pls.score(X_test, y_test)) < 1e-10

quit()
if (True):
    fig1, axes1 = plt.subplots(1, 2)
    fig2, axes2 = plt.subplots(1, 2)

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

        ncomp = np.array(ncomp)
        axes1[0].plot(ncomp, pls,  label=f"center={center}, scale={scale}")
        axes1[1].plot(ncomp, opls, label=f"center={center}, scale={scale}")

        axes2[0].plot(ncomp-((not center) and (not scale)), pls,
                      label=f"center={center}, scale={scale}")
        axes2[1].plot(ncomp-((not center) and (not scale)), opls,
                      label=f"center={center}, scale={scale}")

        for axes in (axes1, axes2):
            axes[0].set_title("Pure PLS")
            axes[1].set_title("OPLS filtering with one component")

            for ax in axes:
                ax.legend()
                ax.set_xlabel("number of components")
                ax.set_ylabel("Test score")

    for fig in (fig1, fig2):
        fig.set_size_inches(14, 8)
        fig.tight_layout()
    fig1.savefig(pathlib.Path(__file__).parent / "center_scale_test1.png")
    fig2.savefig(pathlib.Path(__file__).parent / "center_scale_test2.png")

    # quit()

if (True):
    maxk = 15

    ncomp = np.arange(maxk)+1
    pls_score = []
    for k in ncomp:
        print(k, end="\r")
        pls = PLS(n_components=k).fit(X_train, y_train)
        pls_score.append(pls.score(X_test, y_test))

    print("pls ", ["%.4f" % v for v in pls_score])

    ncomp = np.arange(maxk)+1
    pls_score = []
    pls = PLS(n_components=maxk).fit(X_train, y_train)
    for k in ncomp:
        print(k, end="\r")
        pls_score.append(pls.score(X_test, y_test, ndim=k))

    print("pls ", ["%.4f" % v for v in pls_score])

    opls_pls_score = {}
    ncomp_opls = np.arange(1, 7)
    for i in ncomp_opls:
        opls_pls_score[i] = []
        opls = OPLS(n_components=i).fit(X_train, y_train)
        pls = PLS(n_components=maxk).fit(opls.correct(X_train), y_train)
        for k in ncomp:
            print(i, k, end="\r")
            opls_pls_score[i].append(
                pls.score(opls.correct(X_test), y_test, ndim=k))

        print("%2d  " % i, ["%.4f" % v for v in opls_pls_score[i]])

    fig, ax = plt.subplots(1)
    ax.plot(ncomp, pls_score, label="PLS")
    for i in opls_pls_score:
        ax.plot(ncomp, opls_pls_score[i], "--", label=f"OPLS({i})-PLS")

    ax.legend()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(pathlib.Path(__file__).parent / "scores.png")

    fig, ax = plt.subplots(1)
    ax.plot(ncomp, pls_score, label="PLS")
    for i in opls_pls_score:
        ax.plot(ncomp+i, opls_pls_score[i], "--", label=f"OPLS({i})-PLS")

    ax.set_xlim(ncomp.min()-1, ncomp.max()+1)
    ax.legend()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(pathlib.Path(__file__).parent / "scores2.png")

jaas = 10
print(0, jaas,
      PLS(n_components=10).fit(X_train, y_train).score(X_test, y_test))
for i in range(1, jaas):
    k = jaas-i
    opls = OPLS(n_components=i).fit(X_train, y_train)
    pls = PLS(n_components=k).fit(opls.correct(X_train), y_train)
    print(i, k, pls.score(opls.correct(X_test), y_test),
          OPLS_PLS(i, k).fit(X_train, y_train).score(X_test, y_test))
