
from scipy.optimize import differential_evolution
import numpy as np
from BIAS import BIAS, f0
from tqdm import tqdm
from bayes_opt.bayes_optim import BO, RealSpace, AnnealingBO
from bayes_opt.bayes_optim.extension import PCABO
from bayes_opt.bayes_optim.surrogate import GaussianProcess, RandomForest, s0



dim = 30
num_samples = 600
space = RealSpace([0, 1]) * dim  # create the search space
thetaL = 1e-10 * (1 - 0) * np.ones(dim)
thetaU = 10 * (1 - 0) * np.ones(dim)


#do 30 independent runs (5 dimensions)


for bo_choice in ["BO"]: #"BO", "AnnealingBO", "PCABO"
    for model_choice in ["GP", "RF"]:#
        for aq_choice in ["MGFI", "UCB", "EI", "EpsilonPI"]:
            for opt_choice in ["BFGS"]:           #"MIES", "OnePlusOne_Cholesky_CMA",       
                samples = []
                print(f"Evaluating {bo_choice} with {model_choice} and {aq_choice} and {opt_choice}")
                for i in tqdm(np.arange(num_samples)):
                    np.random.seed(i)
                    doe_size = 100
                    max_FEs = 1000
                    if model_choice == "GP":
                        model = GaussianProcess(                # create the GPR model
                            thetaL=thetaL, thetaU=thetaU
                        )
                    elif model_choice == "RF":
                        model = RandomForest(levels=space.levels)
                    else:
                        model = s0()
                        doe_size = 1
                        max_FEs = 10

                    if bo_choice == "BO":
                        opt = BO(
                            search_space=space,
                            obj_fun=f0,
                            model=model,
                            DoE_size=doe_size,                         # number of initial sample points
                            max_FEs=doe_size+1,                         # maximal function evaluation
                            verbose=False,
                            acquisition_fun=aq_choice,
                            acquisition_optimization={"optimizer": opt_choice, 'max_FEs': max_FEs}
                        )
                    elif bo_choice == "PCABO":
                        opt = PCABO(
                            search_space=space,
                            obj_fun=f0,
                            model=model,
                            DoE_size=doe_size,                         # number of initial sample points
                            max_FEs=doe_size+1,                         # maximal function evaluation
                            verbose=False,
                            n_components=0.95,
                            acquisition_fun=aq_choice,
                            acquisition_optimization={"optimizer": opt_choice, 'max_FEs': max_FEs}
                        )
                    xopt, fopt, stop_dict = opt.run()
                    samples.append(xopt)

                samples = np.array(samples)
                np.save(f"bo/exp_{bo_choice}-{model_choice}-{aq_choice}-{opt_choice}.npy", samples)