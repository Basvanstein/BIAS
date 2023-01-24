#example of using the BIAS toolbox to test a DE algorithm

from scipy.optimize import differential_evolution
import numpy as np
from BIAS import BIAS, f0, install_r_packages

install_r_packages()

bounds = [(0,1), (0, 1), (0, 1), (0, 1), (0, 1)]

#do 30 independent runs (5 dimensions)
samples = []
print("Performing optimization method 50 times of f0.")
for i in np.arange(50):
    result = differential_evolution(f0, bounds, maxiter=100)
    samples.append(result.x)

samples = np.array(samples)

test = BIAS()
# use the classical stastistical approach to detect BIAS
print(test.predict(samples, show_figure=True))

#use the trained deep learning model to predict and explain BIAS
y, preds = test.predict_deep(samples)
test.explain(samples, preds, filename="explanation.png")
