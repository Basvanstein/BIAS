import sys
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    adjusted_mutual_info_score,
    pairwise_distances,
    pairwise_distances_argmin_min,
)

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

importr('data.table')
importr('goftest')
ddst = importr('ddst')
pwr = importr('PoweR')

robjects.r('''
R_test_ad <- function(x, max_=1) {
    return(ad.test(x, "punif", max=max_, min=0)[[2]])
}

R_test_norm <- function(x, test='Shapiro') {
    qnorm_temp <- qnorm(x)
    qnorm_temp[is.infinite(qnorm_temp)] <- 4*sign(qnorm_temp[is.infinite(qnorm_temp)])
    if (test == 'Shapiro') {
        return(shapiro.test(qnorm_temp)[[2]])
    } else {
        return(AutoSEARCH::jb.test(qnorm_temp)$p.value)
    }
}
''')

def get_mi(X, type_="med"):
    mutuals = []
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            if i != j:
                mutuals.append(
                    mutual_info_regression(X[:, i].reshape(-1, 1), X[:, j])[0]
                )
    if type_ == "med":
        return np.median(mutuals)
    return np.max(mutuals)


def get_mmpd(X):
    pairwisedistances = []
    #     print(len(X))
    for i in range(len(X)):
        one = X[i]
        rest = np.append(X[:i], X[i + 1 :], axis=0)
        res, res_dist = pairwise_distances_argmin_min(one.reshape(1, -1), rest)
        pairwisedistances.append(res_dist[0])
    return np.max(pairwisedistances)


def get_mddlud(X, type_="med"):
    baseline_space = np.linspace(0, 1, len(X))
    lindist = []
    for i in range(X.shape[1]):
        lindist.append(np.max(np.abs(np.sort(X[:, i]) - baseline_space)))
    if type_ == "med":
        return np.median(lindist)
    return max(lindist)


def is_valid(x, centers, gap_size):
    for c in centers:
        if np.abs(x - c) < gap_size:
            return False
    return True


def get_simulated_data(scen, rep=1000, n_samples=100, kwargs={}):
    if scen == "unif":
        data_arr = np.random.uniform(size=(n_samples, rep))
    elif scen == "trunc_unif":
        min_ = kwargs["min"]
        max_ = kwargs["max"]
        data_arr = np.random.uniform(size=(n_samples, rep), low=min_, high=max_)
    elif scen == "spikes":
        data_arr = (
            np.random.randint(kwargs["max"] + 1, size=(n_samples, rep)) / kwargs["max"]
        )
    elif scen == "shifted_spikes":
        possible_vals = [i / kwargs["max"] for i in range(kwargs["max"] + 1)]
        translations = np.random.normal(0, kwargs["sigma"], size=(kwargs["max"] + 1))
        possible_vals = [
            i + j if 0 <= i + j <= 1 else i for i, j in zip(possible_vals, translations)
        ]
        data_arr = np.random.choice(possible_vals, size=(n_samples, rep))
    elif scen == "norm":
        nr_req = n_samples * rep
        data_temp = np.random.normal(kwargs["mu"], kwargs["sigma"], size=(nr_req * 10))
        data_arr = [x for x in data_temp if 0 < x < 1][:nr_req]
        data_arr = np.array(data_arr).reshape((n_samples, rep))
    elif scen == "cauchy":
        nr_req = n_samples * rep
        data_temp = ss.cauchy.rvs(kwargs["mu"], kwargs["sigma"], size=(nr_req * 10))
        data_arr = [x for x in data_temp if 0 < x < 1][:nr_req]
        data_arr = np.array(data_arr).reshape((n_samples, rep))
    elif scen == "inv_cauchy":
        nr_req = n_samples * rep
        data_temp = ss.cauchy.rvs(kwargs["mu"], kwargs["sigma"], size=(nr_req * 10))
        data_arr = [x for x in data_temp if 0 < x < 1][:nr_req]
        data_arr = [
            1 + kwargs["mu"] - x if x > kwargs["mu"] else kwargs["mu"] - x
            for x in data_arr
        ]  # Not efficient, but works. Maybe improve later
        data_arr = np.array(data_arr).reshape((n_samples, rep))
    elif scen == "inv_norm":
        nr_req = n_samples * rep
        data_temp = np.random.normal(kwargs["mu"], kwargs["sigma"], size=(nr_req * 10))
        data_arr = [x for x in data_temp if 0 < x < 1][:nr_req]
        data_arr = [
            1 + kwargs["mu"] - x if x > kwargs["mu"] else kwargs["mu"] - x
            for x in data_arr
        ]  # Not efficient, but works. Maybe improve later
        data_arr = np.array(data_arr).reshape((n_samples, rep))
    elif scen == "gaps":
        temp = []
        for rep in range(rep):
            data_temp = np.random.uniform(size=(n_samples * 10))
            centers = np.random.uniform(size=kwargs["n_centers"])
            temp.append(
                [x for x in data_temp if is_valid(x, centers, kwargs["sigma"])][
                    :n_samples
                ]
            )
        data_arr = np.array(temp).transpose()
    elif scen == "consistent_gaps":
        temp = []
        centers = np.random.uniform(size=kwargs["n_centers"])
        for rep in range(rep):
            data_temp = np.random.uniform(size=(n_samples * 10))
            temp.append(
                [x for x in data_temp if is_valid(x, centers, kwargs["sigma"])][
                    :n_samples
                ]
            )
        data_arr = np.array(temp).transpose()
    elif scen == "clusters":
        temp = []
        for rep in range(rep):
            centers = np.random.uniform(size=kwargs["n_centers"])
            samples = [
                np.random.normal(loc=x, scale=kwargs["sigma"])
                for x in np.random.choice(centers, size=n_samples * 10)
            ]
            temp.append([x for x in samples if 0 < x < 1][:n_samples])
        data_arr = np.array(temp).transpose()
    elif scen == "consistent_clusters":
        temp = []
        centers = np.random.uniform(size=kwargs["n_centers"])
        for rep in range(rep):
            samples = [
                np.random.normal(loc=x, scale=kwargs["sigma"])
                for x in np.random.choice(centers, size=n_samples * 10)
            ]
            temp.append([x for x in samples if 0 < x < 1][:n_samples])
        data_arr = np.array(temp).transpose()
    elif scen == "part_unif":
        temp = []
        for rep in range(rep):
            n_unif = int(np.ceil(kwargs["frac_unif"] * n_samples))
            data_temp = np.random.uniform(size=(n_unif))
            new_points = [
                np.random.normal(loc=x, scale=kwargs["sigma"])
                for x in np.random.choice(data_temp, size=n_samples * 10)
            ]
            data_new = [x for x in new_points if 0 < x < 1][:n_samples]
            #             deviations = np.random.normal(size = len(data_temp), scale=kwargs['sigma'])
            #             new_points = [x + y for x,y in zip(data_temp, deviations) if 0 < x+y < 1]
            data_new = np.append(data_temp[:n_unif], data_new[: (n_samples - n_unif)])
            temp.append(data_new)
        data_arr = np.array(temp).transpose()
    elif scen == "bound_thing":
        temp = []
        for _ in range(rep):
            n_01 = int(np.ceil((1 - kwargs["frac_between"]) * n_samples))
            data_temp = np.random.uniform(size=(n_samples))
            data_temp[
                np.random.choice(range(n_samples), n_01, replace=False)
            ] = np.random.choice(
                [0, 1], size=n_01, p=[kwargs["frac_0"], 1 - kwargs["frac_0"]]
            )
            #             for idx in np.random.randint(0, n_samples, n_01):
            #                 data_temp[idx] = np.random.csv"
            temp.append(np.array(data_temp))
        data_arr = np.array(temp).transpose()
    return data_arr


scenario_dict = {
    "unif": None,
    "trunc_unif": ["min", "max"],
    "spikes": ["max"],
    "shifted_spikes": ["max", "sigma"],
    "norm": ["sigma", "mu"],
    "inv_norm": ["sigma", "mu"],
    "cauchy": ["sigma", "mu"],
    "inv_cauchy": ["sigma", "mu"],
    "gaps": ["n_centers", "sigma"],
    "clusters": ["n_centers", "sigma"],
    "part_unif": ["frac_unif", "sigma"],
}

scenario_dict_across = {
    "unif": None,
    "trunc_unif": ["min", "max"],
    "spikes": ["max"],
    "shifted_spikes": ["max", "sigma"],
    "norm": ["sigma", "mu"],
    "inv_norm": ["sigma", "mu"],
    "cauchy": ["sigma", "mu"],
    "inv_cauchy": ["sigma", "mu"],
    "gaps": ["n_centers", "sigma"],
    "consistent_gaps": ["n_centers", "sigma"],
    "clusters": ["n_centers", "sigma"],
    "consistent_clusters": ["n_centers", "sigma"],
    "part_unif": ["frac_unif", "sigma"],
}

# Note: this file is set up terribly (since it is derived from my notebook-code). TODO: Figure out a better way to structure this!!!


def get_test_dict(n_samples, per_dim=True):
    ### Start by setting up the reference values which need to be gotten from simulations ###

    # spacing-values
    dist_vals_rand = np.array(
        [
            np.diff(np.sort(np.append(np.random.uniform(size=n_samples), [0, 1])))
            for _ in range(1000)
        ]
    ).reshape(-1)
    dist_vals_rand2 = []
    for rep in range(1000):
        x = np.sort(np.append(np.random.uniform(size=n_samples), [0, 1]))
        dist_vals_rand2.append(x[2:] - x[:-2])
    dist_vals_rand2 = np.array(dist_vals_rand2).reshape(-1)

    dist_vals_rand3 = []
    for rep in range(1000):
        x = np.sort(np.append(np.random.uniform(size=n_samples), [0, 1]))
        dist_vals_rand3.append(x[3:] - x[:-3])
    dist_vals_rand3 = np.array(dist_vals_rand3).reshape(-1)

    #     #Range values
    #     dists = [np.max(x) - np.min(x) for x in np.random.uniform(size=(10000,n_samples))]
    #     mins = [np.min(x) for x in np.random.uniform(size=(10000,n_samples))]
    #     maxs = [np.max(x) for x in np.random.uniform(size=(10000,n_samples))]

    #     #linspace baseline
    comp_to = np.linspace(0, 1, num=n_samples)
    #     wassersteins = [np.sum(np.abs(np.sort(x) - comp_to)) for x in np.random.uniform(size=(10000,n_samples))]
    #     lindist_min = [np.min(np.abs(np.sort(x) - comp_to)) for x in np.random.uniform(size=(10000,n_samples))]
    #     lindist_max = [np.max(np.abs(np.sort(x) - comp_to)) for x in np.random.uniform(size=(10000,n_samples))]

    #     #max pairwise distances
    #     max_pair_dists = [np.max(np.diff(np.sort(np.random.uniform(size=(n_samples))))) for x in range(10000)]

    ### Define the tests. For now, this is needed here, since it relies on the simulated reference values :(
    def test_spacing(x, m=1, alpha=0.01):
        x = np.sort(np.append(x, [0, 1]))
        if m == 1:
            p = ss.ks_2samp(np.diff(x), dist_vals_rand)[1]
        elif m == 2:
            p = ss.ks_2samp(x[2:] - x[:-2], dist_vals_rand2)[1]
        else:
            p = ss.ks_2samp(x[3:] - x[:-3], dist_vals_rand3)[1]
        return p  # < alpha

    def test_range(x, alpha=0.01):
        return np.max(x) - np.min(x)  # <= np.quantile(dists, alpha)

    def test_edges(x, type_="min", alpha=0.01):
        if type_ == "min":
            return np.min(x)  # >= np.quantile(mins, 1-alpha)
        else:
            return np.max(x)  # <= np.quantile(maxs, alpha)

    def test_ad(x, transform=False, alpha=0.01):
        if transform:
            x = np.abs(x - 0.5)
            return robjects.globalenv["R_test_ad"](robjects.FloatVector(x), 0.5)[
                0
            ]  # < alpha
        return robjects.globalenv["R_test_ad"](robjects.FloatVector(x))[0]  # < alpha

    def test_normal_transformed(x, test="Shapiro", alpha=0.01):
        return robjects.globalenv["R_test_norm"](robjects.FloatVector(x), test)[
            0
        ]  # < alpha

    # TODO: fix the naming scheme (this is mddlud!)
    def test_lindist_dim(x, type_="min", alpha=0.01):
        if type_ == "max":
            return np.max(
                np.abs(np.sort(x) - comp_to)
            )  # >= np.quantile(lindist_max, 1-alpha)
        else:
            return np.min(
                np.abs(np.sort(x) - comp_to)
            )  # <= np.quantile(lindist_min, alpha)

    def test_pairwise_dists_dim(x, type_="min", alpha=0.01):
        if type_ == "max":
            return np.max(
                np.diff(np.sort(x))
            )  # >= np.quantile(max_pair_dists, 1-alpha)
        else:
            return np.max(np.diff(np.sort(x)))  # <= np.quantile(max_pair_dists, alpha)

    def test_kurtosis(x, alpha=0.01):
        return ss.kurtosis(ss.norm.ppf(x))

    #         return not (np.quantile(kurts,alpha/2) < ss.kurtosis(ss.norm.ppf(x)) < np.quantile(kurts,1-alpha/2))

    def test_wasserstein(x, alpha=0.01):
        # Note: not scaled for sample size (won't matter for result, but need to keep in mind that right baseline needs to be used!)
        return np.sum(
            np.abs(np.sort(x) - comp_to)
        )  # > np.quantile(wassersteins,1-alpha)

    def test_ddst(x, alpha=0.01):
        return list(ddst.ddst_uniform_test(robjects.FloatVector(x), compute_p=True))[
            -1
        ][
            0
        ]  # < alpha

    def test_pwr(x, test_nr, alpha=0.01):
        return pwr.statcompute(test_nr, robjects.FloatVector(x))[0][0]

    test_battery_per_dim = {
        "1-spacing": test_spacing,
        "2-spacing": partial(test_spacing, m=2),
        "3-spacing": partial(test_spacing, m=3),
        "range": test_range,
        "min": test_edges,
        "max": partial(test_edges, type_="max"),
        "ad": test_ad,
        "ad_transform": partial(test_ad, transform=True),
        "shapiro": test_normal_transformed,
        "jb": partial(test_normal_transformed, test="jb"),
        "mdd_min": test_lindist_dim,
        "mdd_max": partial(test_lindist_dim, type_="max"),
        "kurtosis": test_kurtosis,
        "mmpd_max": test_pairwise_dists_dim,
        "mmpd_min": partial(test_pairwise_dists_dim, type_="max"),
        "wasserstein": test_wasserstein,
        "ddst": test_ddst,
        "kolmogorov": partial(test_pwr, test_nr=63),
        "CvM": partial(test_pwr, test_nr=64),
        "AD_pwr": partial(test_pwr, test_nr=65),
        "Durbin": partial(test_pwr, test_nr=66),
        "Kuiper": partial(test_pwr, test_nr=67),
        "HG1": partial(test_pwr, test_nr=68),
        "HG2": partial(test_pwr, test_nr=69),
        "Greenwood": partial(test_pwr, test_nr=70),
        "QM": partial(test_pwr, test_nr=71),
        "RC": partial(test_pwr, test_nr=72),
        "Moran": partial(test_pwr, test_nr=73),
        "Cressie1": partial(test_pwr, test_nr=74),
        "Cressie2": partial(test_pwr, test_nr=75),
        "Vasicek": partial(test_pwr, test_nr=76),
        "Swartz": partial(test_pwr, test_nr=77),
        "Morales": partial(test_pwr, test_nr=78),
        "Pardo": partial(test_pwr, test_nr=79),
        "Marhuenda": partial(test_pwr, test_nr=80),
        "Zhang1": partial(test_pwr, test_nr=81),
        "Zhang2": partial(test_pwr, test_nr=82),
    }

    if per_dim:
        return test_battery_per_dim

    def test_mi(X, type_="med", alpha=0.01):
        mi = get_mi(X, type_)
        if type_ == "med":
            return mi  # > np.quantile(med_mis, 1-alpha)
        return mi  # > np.quantile(max_mis, 1-alpha)

    def test_mmpd(X, alpha=0.01):
        mmpd = get_mmpd(X)
        return mmpd  # > np.quantile(mmpds, 1-alpha)

    def test_mddlud(X, type_="med", alpha=0.01):
        mddlud = get_mddlud(X, type_)
        if type_ == "med":
            return mddlud  # > np.quantile(med_ddluds, 1-alpha)
        return mddlud  # > np.quantile(max_ddluds, 1-alpha)

    def test_spacing_across(X, m=1, alpha=0.01):
        # Not very efficient, but works for now
        diffs = []
        for dim in range(X.shape[1]):
            x = np.sort(np.append(X[:, dim], [0, 1]))
            if m == 1:
                diffs.append(np.diff(x))
            else:
                diffs.append(x[m:] - x[: (-1 * m)])
        diffs = np.array(diffs).reshape(-1)
        if m == 1:
            p = ss.ks_2samp(diffs, dist_vals_rand)[1]
        elif m == 2:
            p = ss.ks_2samp(diffs, dist_vals_rand2)[1]
        else:
            p = ss.ks_2samp(diffs, dist_vals_rand3)[1]
        return p  # < alpha

    test_battery_across_dim = {
        "mi": test_mi,
        #     'max_mi' : partial(test_mi, type_='max'),
        "mmpd": test_mmpd,
        "med_ddlud": test_mddlud,
        "max_ddlud": partial(test_mddlud, type_="max"),
    }

    def run_test_aggr(x, test, **kwargs):
        y = x.reshape(-1)
        return test(y, **kwargs)

    test_battery_aggr = {}
    for k, v in test_battery_per_dim.items():
        if (
            "mdd" not in k
            and "mmpd" not in k
            and "spacing" not in k
            and "wasser" not in k
        ):
            test_battery_aggr[k] = partial(run_test_aggr, test=v)

    test_battery_aggr["1-spacing"] = test_spacing_across
    test_battery_aggr["2-spacing"] = partial(test_spacing_across, m=2)
    test_battery_aggr["3-spacing"] = partial(test_spacing_across, m=3)
    test_battery_aggr.pop("range", None)
    test_battery_aggr.pop("min", None)
    test_battery_aggr.pop("max", None)

    return {**test_battery_across_dim, **test_battery_aggr}


def runParallelFunction(runFunction, arguments):
    """
    Return the output of runFunction for each set of arguments,
    making use of as much parallelization as possible on this system

    :param runFunction: The function that can be executed in parallel
    :param arguments:   List of tuples, where each tuple are the arguments
                        to pass to the function
    :return:
    """

    arguments = list(arguments)
    p = Pool(min(cpu_count(), len(arguments)))
    #     local_func = partial(func_star, func=runFunction)
    results = p.map(runFunction, arguments)
    p.close()
    return results


def run_scenario_across(
    scen_list, foldername="", rep=100, dims=30, alpha=0.01, n_samples=100
):
    np.random.seed(42)
    records = {}
    kwargs = scen_list[1]
    scen = scen_list[0]
    test_battery_across = get_test_dict(n_samples, False)
    if n_samples > 150:
        test_battery_across.pop("jb", None)
        test_battery_across.pop("shapiro", None)
    for r in range(rep):
        data_arr = get_simulated_data(scen, dims, n_samples, kwargs)
        #         print(data_arr.shape)
        for tname, tfunc in test_battery_across.items():
            print(tname)
            if tname in records:
                records[tname].append(tfunc(data_arr, alpha=alpha))
            else:
                records[tname] = [tfunc(data_arr, alpha=alpha)]
    dt = pd.DataFrame.from_dict(records)
    scen_name = f"{foldername}S{scen}"
    for k, v in kwargs.items():
        dt[f"{k}_"] = v
        scen_name = f"{scen_name}_{k}_{v}"
    scen_name = f"{scen_name}.csv"
    dt["scen"] = scen
    dt["n_samples"] = n_samples
    dt["dims"] = dims
    dt.to_csv(scen_name)


#     return dt


def run_scenario(scen_list, foldername="", rep=1500, alpha=0.01, n_samples=100):
    np.random.seed(42)
    kwargs = scen_list[1]
    scen = scen_list[0]
    #     print(scen)
    data_arr = get_simulated_data(scen, rep, n_samples, kwargs)
    records = {}
    test_battery_per_dim = get_test_dict(n_samples)
    for tname, tfunc in test_battery_per_dim.items():
        print(tname)
        temp = []
        for r in range(rep):
            temp.append(tfunc(data_arr[:, r], alpha=alpha))
        records[tname] = temp
    dt = pd.DataFrame.from_dict(records)
    scen_name = f"{foldername}S{scen}"
    for k, v in kwargs.items():
        dt[f"{k}_"] = v
        scen_name = f"{scen_name}_{k}_{v}"
    scen_name = f"{scen_name}.csv"
    #     print(scen_name)
    dt["scen"] = scen
    dt["n_samples"] = n_samples
    dt.to_csv(scen_name)


#     return dt


def get_scens_per_dim():
    scens = [["unif", {}]]
    for temp in [0.025, 0.05, 0.1, 0.2]:
        scens.append(["trunc_unif", {"min": temp / 2, "max": 1 - temp / 2}])
    for temp in [0.025, 0.05, 0.1, 0.2]:
        scens.append(["trunc_unif", {"min": temp, "max": 1}])
    for max_ in [25, 50, 100, 150, 200, 250]:
        scens.append(["spikes", {"max": max_}])
    for max_ in [25, 50, 100, 150, 200, 250]:
        for sigma in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
            scens.append(["shifted_spikes", {"max": max_, "sigma": sigma}])
    for s in ["norm", "inv_norm", "cauchy", "inv_cauchy"]:
        for sigma in [0.1, 0.2, 0.3, 0.4]:
            for mu in [0.5, 0.6, 0.7]:
                scens.append([s, {"sigma": sigma, "mu": mu}])
    for n_centers in [1, 2, 3, 4, 5]:
        for gap_rad in [0.01, 0.02, 0.03, 0.04, 0.05]:
            scens.append(["gaps", {"n_centers": n_centers, "sigma": gap_rad}])
    for n_centers in [1, 2, 3, 4, 5]:
        for gap_rad in [0.01, 0.025, 0.05, 0.1, 0.2, 0.3]:
            scens.append(["clusters", {"n_centers": n_centers, "sigma": gap_rad}])
    for n_unif in [0.1, 0.25, 0.5]:
        for sigma in [0.01, 0.02, 0.05, 0.1]:
            scens.append(["part_unif", {"frac_unif": n_unif, "sigma": sigma}])
    for f_0 in [0.1, 0.35, 0.45, 0.5]:
        for f_between in [0.5, 0.25, 0.1, 0.05, 0.025, 0.01]:
            scens.append(["bound_thing", {"frac_between": f_between, "frac_0": f_0}])
    return scens


def get_scens_across_dim():
    scens = [["unif", {}]]
    for temp in [0.01, 0.025, 0.05, 0.1, 0.2]:
        scens.append(["trunc_unif", {"min": temp / 2, "max": 1 - temp / 2}])
        scens.append(["trunc_unif", {"min": temp, "max": 1}])
    for max_ in [25, 50, 100, 150, 200, 250, 500, 1000]:
        scens.append(["spikes", {"max": max_}])
        for sigma in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
            scens.append(["shifted_spikes", {"max": max_, "sigma": sigma}])
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for mu in [0.5, 0.6, 0.7]:
            scens.append(["norm", {"sigma": sigma, "mu": mu}])
            scens.append(["inv_norm", {"sigma": sigma, "mu": mu}])
            scens.append(["cauchy", {"sigma": sigma, "mu": mu}])
            scens.append(["inv_cauchy", {"sigma": sigma, "mu": mu}])
    for n_centers in [1, 2, 3, 4, 5]:
        for gap_rad in [0.01, 0.02, 0.03, 0.04, 0.05]:
            scens.append(
                ["consistent_gaps", {"n_centers": n_centers, "sigma": gap_rad}]
            )
            scens.append(["gaps", {"n_centers": n_centers, "sigma": gap_rad}])
    for n_centers in [1, 2, 3, 4, 5]:
        for gap_rad in [0.01, 0.025, 0.05, 0.1, 0.2, 0.3]:
            scens.append(
                ["consistent_clusters", {"n_centers": n_centers, "sigma": gap_rad}]
            )
            scens.append(["clusters", {"n_centers": n_centers, "sigma": gap_rad}])
    for n_unif in [0.1, 0.25, 0.5]:
        for sigma in [0.01, 0.02, 0.05, 0.1]:
            scens.append(["part_unif", {"frac_unif": n_unif, "sigma": sigma}])
    return scens


def get_scens_inv():
    # Get only the inv-based scenarios
    scens = []
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for mu in [0.6, 0.7]:
            scens.append(["inv_norm", {"sigma": sigma, "mu": mu}])
            scens.append(["inv_cauchy", {"sigma": sigma, "mu": mu}])
    return scens


def get_scens_bound():
    # Get only the added heavy-bound scenario
    scens = []
    for f_0 in [0.1, 0.35, 0.45, 0.5]:
        for f_between in [0.5, 0.25, 0.1, 0.05, 0.025, 0.01]:
            scens.append(["bound_thing", {"frac_between": f_between, "frac_0": f_0}])
    return scens


def run_test_cases(n_samples, fname="Datatables", per_dim=True, rep=1500):
    if per_dim:
        scens = get_scens_per_dim()
        foldername = f"{fname}/S{n_samples}/"
        partial_run = partial(
            run_scenario, foldername=foldername, n_samples=n_samples, rep=rep
        )
        runParallelFunction(partial_run, scens)
    else:
        scens = get_scens_across_dim()
        foldername = f"{fname}/S{n_samples}_Across/"
        partial_run = partial(
            run_scenario_across, foldername=foldername, n_samples=n_samples
        )
        runParallelFunction(partial_run, scens)


if __name__ == "__main__":
    idx_nr = int(sys.argv[1])
    #     rep = int(sys.argv[2])

    # idx_nr decides which experiment is run (division on nodes)
    fname = "/var/scratch/dlvermet/SB"
    #     fname = "Datatables"
    run_per = idx_nr < 4
    s = [30, 50, 100, 600][idx_nr % 4]
    #     for s in [30, 50, 100, 600]:
    run_test_cases(s, fname, run_per)
#     run_test_cases(100)
#     alpha = [0.05, 0.01, ]
