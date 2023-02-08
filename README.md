# OPLS #

OPLS and PLS regressors

## Installation

Run

```
pip install OPLS-MD
```

or to get the latest development commits clone the github repo and run

```
pip install .
```

## Usage

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
from OPLS import PLS, OPLS

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


## References

Main references for OPLS:
    
    [1] Wold S, et al. PLS-regression: a basic tool of chemometrics.
        Chemometr Intell Lab Sys 2001, 58, 109–130.
    [2] Bylesjo M, et al. Model Based Preprocessing and Background
        Elimination: OSC, OPLS, and O2PLS. in Comprehensive Chemometrics.
