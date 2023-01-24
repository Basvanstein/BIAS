import numpy as np
from BIAS import BIAS, f0, install_r_packages
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

names = ['SMAC4HPO', 'SMAC4BB']
for smac_name in names:
    samples = np.load(f"bo/exp_smac-{smac_name}.npy")
    test = BIAS()
    # use the classical stastistical approach to detect BIAS
    dt_rejections, _ = test.predict(samples, show_figure=False)
    np.save(f"bo/rejections_{smac_name}.npy", dt_rejections)
    test.plot_swarm_with_heatmap(samples, dt_rejections, f"bo/fig_smac-{smac_name}.png")

    #use the trained deep learning model to predict and explain BIAS
    y, preds = test.predict_deep(samples)
    np.save(f"bo/preds_{smac_name}.npy", preds)
    test.explain(samples, preds, filename=f"bo/deep_smac-{smac_name}.png")

for bo_choice in ["BO"]: #"BO", "AnnealingBO", "PCABO"
    for model_choice in ["GP", "RF"]:#
        for aq_choice in ["MGFI", "UCB", "EI", "EpsilonPI"]:
            for opt_choice in ["MIES", "OnePlusOne_Cholesky_CMA", "BFGS"]:           
                if opt_choice == "BFGS" and model_choice != "GP":
                    continue #      "BFGS" only with GP and RF
                samples = np.load(f"bo/exp_{bo_choice}-{model_choice}-{aq_choice}-{opt_choice}.npy")
                test = BIAS()
                # use the classical stastistical approach to detect BIAS
                dt_rejections, _ = test.predict(samples, show_figure=False)
                test.plot_swarm_with_heatmap(samples, dt_rejections, f"bo/fig_{bo_choice}-{model_choice}-{aq_choice}-{opt_choice}.png")
                np.save(f"bo/rejections_{bo_choice}-{model_choice}-{aq_choice}-{opt_choice}.npy", dt_rejections)

                #use the trained deep learning model to predict and explain BIAS
                y, preds = test.predict_deep(samples)
                np.save(f"bo/preds_{bo_choice}-{model_choice}-{aq_choice}-{opt_choice}.npy", preds)
                test.explain(samples, preds, filename=f"bo/deep_{bo_choice}-{model_choice}-{aq_choice}-{opt_choice}.png")


