import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
from tqdm import tqdm
from smac.initial_design.random_configuration_design import RandomConfigurations

dim = 30
num_samples = 600

class f0:
    @property
    def configspace(self) -> ConfigurationSpace:
        dim = 30
        cs = ConfigurationSpace(seed=0)
        for i in range(dim):
            x = Float(f"x{i}", (0, 1), default=0.5)
            cs.add_hyperparameters([x])
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        return np.random.uniform() 


names = ['SMAC4HPO', 'SMAC4BB']
names_i = 0
for facade in [SMAC4HPO, SMAC4BB]:
    samples = []
    for i in tqdm(np.arange(num_samples)):
        np.random.seed(i)
        model = f0()
        # SMAC scenario object
        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternative to runtime)
                "runcount-limit": 120,  # max duration to run the optimization (in seconds)
                "cs": model.configspace,  # configuration space
                "deterministic": "false",
                "limit_resources": True,  # Uses pynisher to limit memory and runtime
                # Alternatively, you can also disable this.
                # Then you should handle runtime and memory yourself in the TA
                "cutoff": 500,  # runtime limit for target algorithm
                "memory_limit": 3072,  # adapt this to reasonable value for your hardware
            }
        )
        smac = facade(
            scenario=scenario,
            rng=np.random.RandomState(i),
            tae_runner=model.train,
        )
        incumbent = smac.optimize()
        xopt = []
        for j in range(dim):
            xopt.append(incumbent[f"x{j}"])
        samples.append(xopt)

    samples = np.array(samples)
    np.save(f"bo/exp_smac-{names[names_i]}.npy", samples)
    names_i += 1