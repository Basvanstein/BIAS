
from scipy.optimize import differential_evolution
import numpy as np
from BIAS import BIAS, f0
from tqdm import tqdm
from bayes_opt.bayes_optim import BO, RealSpace, AnnealingBO
from bayes_opt.bayes_optim.extension import PCABO
from bayes_opt.bayes_optim.surrogate import GaussianProcess, RandomForest, s0

from bayes_opt.bayes_optim.acquisition import OnePlusOne_Cholesky_CMA, MIES, OnePlusOne_CMA

dim = 30
num_samples = 600
space = RealSpace([0, 1]) * dim  # create the search space
#do 30 independent runs (5 dimensions)

optims = ["OnePlusOne_Cholesky_CMA", "MIES", "OnePlusOne_CMA"]
optim_i = 0
for opt_choice in [OnePlusOne_Cholesky_CMA, MIES, OnePlusOne_CMA]:
    print(f"Evaluating {opt_choice}")
    samples = []
    for i in tqdm(np.arange(num_samples)):
        if (optims[optim_i] == "MIES"):
            opt = opt_choice(
                search_space=RealSpace([-5, 5]) * 30,
                obj_func=f0,
                verbose=False,
            )
        else:
            opt = opt_choice(
                search_space=RealSpace([-5, 5]) * 30,
                obj_fun=f0,
                lb=0,
                ub=1,
                sigma0=0.01,
                ftarget=1e-8,
                verbose=False,
            )

        xopt, fopt, stop_dict = opt.run()         
        samples.append(xopt)

    samples = np.array(samples)
    np.save(f"bo/exp_{optims[optim_i]}.npy", samples)
    optim_i += 1