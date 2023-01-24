# BIAS: Bias In Algorithms, Structural
## A toolbox for detecting structural bias in continuous optimization heuristics.

## Setup

This package requires an R-installation to be present, with the following packages installed:
- PoweR
- AutoSEARCH
- nortest
- data.table
- goftest
- ddst

### Detailed setup

1. Download and install R from https://cran.r-project.org/
2. Download this repository (clone or as zip)
3. Create a python virtual env `python -m venv env`
4. Activate the env (in powershell for example: `env/Scripts/Activate.ps1 `)
5. Install dependencies `pip install -r requirements.txt`
6. Checkout the `example.py` to start using the BIAS toolbox.


## Example

```py
#example of using the BIAS toolbox to test a DE algorithm

from scipy.optimize import differential_evolution
import numpy as np
from BIAS import BIAS, f0, install_r_packages

#run first time to install required R packages
install_r_packages()

bounds = [(0,1), (0, 1), (0, 1), (0, 1), (0, 1)]

#do 30 independent runs (5 dimensions)
samples = []
print("Performing optimization method 30 times of f0.")
for i in np.arange(30):
    result = differential_evolution(f0, bounds, maxiter=100)
    samples.append(result.x)

samples = np.array(samples)

test = BIAS()
print(test.predict(samples, show_figure=True))

y, preds = test.predict_deep(samples)
test.explain(samples, preds, filename="explanation.png")
```

## Additional files

Note: The code for generating the RF used to predict the type of bias is included, but the full RF is not. These can be found on zenodo: https://doi.org/10.6084/m9.figshare.16546041.
The RF models will be downloaded automatically the first time the predict function requires them.

