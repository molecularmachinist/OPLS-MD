# OPLS-MD #

OPLS and PLS regressors with some utility to help with molecular dynamics (MD) simulation analysis.

## Requirements

As long as your Python is a correct version, the two other dependencies will be installed automatically by pip.

- `python>=3.10`
- `numpy>=2.0`
- `scipy>=1.0`

## Installation

Run

```
pip install OPLS-MD
```

or to get the latest development commits (if any) clone the github repo and run

```
pip install .
```

## General usage

Given a structure file and trajectory as `struct.pdb` and `traj.xtc` we can e.g. define a function as the distance of two specific C alpha atoms:

```py
import MDAnalysis as mda
import numpy as np

# Make universe and load trajectory to memory
u = mda.Universe("struct.pdb","traj.xtc")
coords = np.array([u.atoms.positions.copy() for ts in u.trajectory])
# Calculate function as distance between res 2 and 150 CA
sel = u.select_atoms("resid 2 150 and name CA")
y = np.linalg.norm(coords[:,sel.indices[0],:]-coords[:,sel.indices[1],:], axis=-1)
# Flatten from N by M by 3 dimensions to N by 3M
X = coords.reshape((coords.shape[0],-1))
```

Now that we have shape N by 3M array of X values and a one dimensional y, we can start running the OPLS

```py
from OPLS_MD import PLS, OPLS

opls = OPLS(n_components=1).fit(X,y)
```

This will fit a one component OPLS model on the data. To get a new X array, where we have filtered out the orthogonal components, we can use the transform function

```py
X_filt = opls.transform(X)
```

Finally to get our model, we can fit the PLS with the filtered data.

```py
pls = PLS(n_components=5).fit(X_filt,y)
```

If we have a new set of data, called `X_new`, we can predict the y values for it as

```py
X_new_filt = opls.transform(X_new)
y_new_predicted = pls.predict(X_new_filt)
```
or if the corresponding `y_new` are known, we can estimate the coefficient of determination (`r2`) with

```py
X_new_filt = opls.transform(X_new)
r2 = pls.score(X_new_filt, y_new)
```

Having to run two separate models one after the other can be tiresome, so the package also includes as `OPLS_PLS` object, which does the above much easier:


```py
from OPLS import OPLS_PLS

# Fitting:
opls_pls = OPLS_PLS(n_components=1,pls_components=5).fit(X,y)

# Predicting:
y_new_predicted = opls_pls.predict(X_new)

# Scoring:
r2 = opls_pls.score(X_new, y_new)
```


## MD-simulation utility

In the previous example we had to flatten the MD coordinates before running the regressor. The three regressors have MD-utility counterparts with a `_MD`-suffix that can take the coordinates as input.
The first two blocks of the example then become

```py
import MDAnalysis as mda
import numpy as np

# Make universe and load trajectory to memory
u = mda.Universe("struct.pdb","traj.xtc")
coords = np.array([u.atoms.positions.copy() for ts in u.trajectory])
# Calculate function as distance between res 2 and 150 CA
sel = u.select_atoms("resid 2 150 and name CA")
y = np.linalg.norm(coords[:,sel.indices[0],:]-coords[:,sel.indices[1],:], axis=-1)


from OPLS_MD import PLS_MD, OPLS_MD

opls = OPLS_MD(n_components=1).fit(coord,y)
```

Instead of a coordinate array, the X can also be given as an MDAnalysis Universe or AtomGroup, with a trajectory length matching the y-array.

## Visualising the model

The regressors include an `inverse_predict`-function, which takes in the y-values and outputs the interpolated corresponding X-structures. This can be useful with univariate y to visualize the coefficent vector of the model. With the `*_MD` variants the output will have the correct shape of `(y.shape[0], natoms, ndim)`. With a trained `PLS_MD` model `pls`, we can write a trajectory with


```py
nsteps=101
u = mda.Universe("struct.pdb")
with mda.Writer("pls_coeff.xtc", u.atoms.n_atoms) as w:
        X_interp = pls.inverse_predict(np.linspace(y.min(),y.max(),nsteps))
        for crd in X_interp:
            u.atoms.positions = crd
            w.write(u)
```

## Testing the number of components

Normally to test the different numbers of components you would simply retrain the models with different numbers. The PLS model saves the coefficients of each previous component up to `n_components`. The `predict`, `inverse_predict` and `score` -functions take `ndim` as an optional argument to set with how many components the calculation should be made.

To get the score over each number of components up to 10 you can run

```py
maxcomp = 10
pls = PLS(n_components=maxcomp).fit(X_train, y_train)
score = []
for k in range(1,maxcomp+1):
    score.append(pls.score(X_test, y_test, ncomp=k))

```

The same works with `PLS_MD` and `OPLS_PLS_MD`. With the latter it only affects the underlying PLS model, not the OPLS filter.


## References

Main references for OPLS:
    
    [1] Wold S, et al. PLS-regression: a basic tool of chemometrics.
        Chemometr Intell Lab Sys 2001, 58, 109–130.
    [2] Bylesjo M, et al. Model Based Preprocessing and Background
        Elimination: OSC, OPLS, and O2PLS. in Comprehensive Chemometrics.
