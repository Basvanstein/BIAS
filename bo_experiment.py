
from scipy.optimize import differential_evolution
import numpy as np
from tqdm import tqdm
from bayes_optim import BO, RealSpace
from bayes_optim.surrogate import GaussianProcess, RandomForest, s0
from mpi4py import MPI


def f0(x):
    """f0 random function, to be used a objective function to test optimization algorithms.

    Args:
        x (list): input for the objective function, ignored since the function is random.

    Returns:
        float: A uniform random number
    """
    return np.random.uniform()


dim = 30
num_samples = 600
space = RealSpace([0, 1]) * dim  # create the search space
thetaL = 1e-10 * (1 - 0) * np.ones(dim)
thetaU = 10 * (1 - 0) * np.ones(dim)


#do 30 independent runs (5 dimensions)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

configurations = []

for bo_choice in ["BO"]: #"BO", "AnnealingBO", "PCABO"
    for model_choice in ["GP", "RF"]:#
        for aq_choice in ["MGFI", "UCB", "EI", "EpsilonPI"]:
            for opt_choice in ["MIES", "OnePlusOne_Cholesky_CMA"]:
                if opt_choice == "BFGS" and model_choice == "RF":
                    continue  # "BFGS" only with GP
                configurations.append((bo_choice, model_choice, aq_choice, opt_choice))

print(len(configurations))
#exit()

bo_choice, model_choice, aq_choice, opt_choice = configurations[rank]

samples = []
print(f"Evaluating {bo_choice} with {model_choice} and {aq_choice} and {opt_choice}")
for i in tqdm(np.arange(num_samples)):
    np.random.seed(i)
    doe_size = 100
    max_FEs = 1000
    update_size = 20
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
            max_FEs=doe_size+update_size,                         # maximal function evaluation
            verbose=False,
            acquisition_fun=aq_choice,
            acquisition_optimization={"optimizer": opt_choice, 'max_FEs': max_FEs}
        )
    xopt, fopt, stop_dict = opt.run()
    samples.append(xopt)

samples = np.array(samples)
np.save(f"bo/exp_{bo_choice}-{model_choice}-{aq_choice}-{opt_choice}.npy", samples)
