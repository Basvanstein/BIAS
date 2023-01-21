
from scipy.optimize import differential_evolution
import numpy as np
from BIAS import BIAS, f0
from tqdm import tqdm
from bayes_opt.bayes_optim import BO, RealSpace, AnnealingBO
from bayes_opt.bayes_optim.extension import PCABO
from bayes_opt.bayes_optim.surrogate import GaussianProcess, RandomForest

np.random.seed(42)

dim = 30
num_samples = 10
space = RealSpace([0, 1]) * dim  # create the search space
thetaL = 1e-10 * (1 - 0) * np.ones(dim)
thetaU = 10 * (1 - 0) * np.ones(dim)


#do 30 independent runs (5 dimensions)


for bo_choice in ["BO"]: #, "AnnealingBO", "PCABO"
    for model_choice in ["GP", "RF"]:
        for aq_choice in ["MGFI", "UCB", "EI", "PI", "EpsilonPI"]:
            samples = []
            print(f"Evaluating {bo_choice} with {model_choice} and {aq_choice}")
            for i in tqdm(np.arange(num_samples)):
                if model_choice == "GP":
                    model = GaussianProcess(                # create the GPR model
                        thetaL=thetaL, thetaU=thetaU
                    )
                else:
                    model = RandomForest()

                if bo_choice == "BO":
                    opt = BO(
                        search_space=space,
                        obj_fun=f0,
                        model=model,
                        DoE_size=100,                         # number of initial sample points
                        max_FEs=1,                         # maximal function evaluation
                        verbose=False,
                        acquisition_fun=aq_choice
                    )
                elif bo_choice == "AnnealingBO":
                    opt = AnnealingBO(
                        search_space=space,
                        obj_fun=f0,
                        model=model,
                        DoE_size=100,                         # number of initial sample points
                        max_FEs=1,                         # maximal function evaluation
                        verbose=False
                    )
                elif bo_choice == "PCABO":
                    opt = PCABO(
                        search_space=space,
                        obj_fun=f0,
                        model=model,
                        DoE_size=100,                         # number of initial sample points
                        max_FEs=1,                         # maximal function evaluation
                        verbose=False
                    )
                xopt, fopt, stop_dict = opt.run()
                samples.append(xopt)

            samples = np.array(samples)
            np.save(f"bo/exp_{bo_choice}-{model_choice}-{aq_choice}.npy", samples)