import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

test_names = [
    "1-spacing",
    "2-spacing",
    "3-spacing",
    "ad",
    "ad_transform",
    "shapiro",
    "jb",
    "ddst",
    "kurtosis",
    "mmpd_min",
    "mmpd_max",
    "range",
    "min",
    "max",
    "mdd_min",
    "mdd_max",
    "wasserstein",
    "kolmogorov",
    "CvM",
    "Durbin",
    "Kuiper",
    "HG1",
    "HG2",
    "Greenwood",
    "QM",
    "RC",
    "Moran",
    "Cressie1",
    "Cressie2",
    "Vasicek",
    "Swartz",
    "Morales",
    "Pardo",
    "Marhuenda",
    "Zhang1",
    "Zhang2",
]

readable_label_dict = {
    "gaps": "Gaps",
    "cauchy": "Center",
    "clusters": "Clusters",
    "inv_cauchy": "Bounds",
    "inv_norm": "Bounds",
    "norm": "Center",
    "part_unif": "Clusters",
    "shifted_spikes": "Discretization",
    "spikes": "Discretization",
    "trunc_unif": "Center",
    "bound_thing": "Bounds",
}


def create_RF_rej(
    included_tests=test_names,
    plot_feat_importance=False,
    use_bias_labels=False,
    feature_order=None,
    rf_file_name=None,
):
    cols_to_get = included_tests + ["scen"]
    dt_samples = pd.DataFrame()
    for sample_size in [30, 50, 100, 600]:
        for f in glob.glob(f"/mnt/e/Research/SB_DE/SB/S{sample_size}/*.csv"):
            dt_temp = pd.read_csv(f)
            #             print(len(dt_temp))
            if dt_temp["scen"][0] != "unif":
                # Remove samples for which no tests reject (non-biased)
                try:
                    if sample_size < 99:
                        dt_rej_temp = pd.read_csv(
                            f"/mnt/e/Research/SB_DE/SB/Rejections/S{sample_size}_A0.01_Cnone_{f[29:]}",
                            index_col=0,
                        )
                    else:
                        dt_rej_temp = pd.read_csv(
                            f"/mnt/e/Research/SB_DE/SB/Rejections/S{sample_size}_A0.01_Cnone_{f[30:]}",
                            index_col=0,
                        )

                    dt_test_only = dt_rej_temp[included_tests]
                    idxs_save = np.where(dt_test_only.transpose().sum() > 0)
                    dt_samples = dt_samples.append(
                        dt_rej_temp[cols_to_get].iloc[idxs_save]
                    )
                except:
                    next
    X = dt_samples[included_tests]
    if use_bias_labels:
        Y = [readable_label_dict[x] for x in dt_samples["scen"]]
    else:
        Y = dt_samples["scen"]

    rf = RandomForestClassifier(oob_score=True, class_weight="balanced")

    rf.fit(X, Y)

    if plot_feat_importance:
        plt.figure(figsize=(19, 10))
        if feature_order is None:
            sbs.barplot(x=included_tests, y=rf.feature_importances_)
        else:
            sbs.barplot(
                x=included_tests, y=rf.feature_importances_, order=feature_order
            )
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"RF_feature_importance.pdf")
        plt.show()

    print(rf.oob_score_)

    if rf_file_name is not None:
        with open(f"{rf_file_name}.pkl", "wb") as output_file:
            pickle.dump(rf, output_file)
    return rf
