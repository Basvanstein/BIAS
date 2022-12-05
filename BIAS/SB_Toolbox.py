from .SB_Test_runner import get_test_dict
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import requests
from io import BytesIO
from zipfile import ZipFile
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from statsmodels.stats.multitest import multipletests
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import seaborn as sbs
import os

pwr = importr('PoweR')

def install_r_packages():
    """Install the required R packages.
    """
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    packnames = ('PoweR', 'AutoSEARCH', 'nortest', 'data.table', 'goftest', 'ddst')
    utils.install_packages(StrVector(packnames))

def f0(x):
    """f0 random function, to be used a objective function to test optimization algorithms.

    Args:
        x (list): input for the objective function, ignored since the function is random.

    Returns:
        float: A uniform random number
    """
    return np.random.uniform()        
class BIAS():
    
    def __init__(self):
        """BIAS toolbox for predicting bias in black box optimization algorithms.
        Predicts both the presence of bias and the bias type. Use f0 as objective function for at least 30 independent optimization runs.

        Args:
            install_r (bool): if set to True, try to install the required R packages automatically.
        """
        self.p_value_columns = ['1-spacing', '2-spacing', '3-spacing','ad', 'ad_transform', 'shapiro', 'jb', 'ddst']


    def _load_ref_vals(self, n_samples, alpha = 0.01, across = False):
        """Helper function to load the reference values needed for calculating the p-values.

        Args:
            n_samples (int): the sample size used for the statistical tests. Can only be 
        in [30,50,100,600]
            alpha (float, optional): Can only be in [0.01, 0.05]. Defaults to 0.01.
            across (bool, optional): Whether we use across dimension reference vals or not. Defaults to False.

        Returns:
            list, list: two lists of reference values loaded from files.
        """
        dirname = os.path.dirname(__file__)
        if across:
            with open(f"{dirname}/data/Crit_vals_across/S{n_samples}_A{alpha}_with_refs.pkl", 'rb') as f:
                ref_vals, _ = pickle.load(f)
            with open(f"{dirname}/data/Crit_vals_pwr_across/S{n_samples}_A{alpha}_with_refs.pkl", 'rb') as f:
                ref_vals_new, _ = pickle.load(f)
        else:
            with open(f"{dirname}/data/Crit_vals/S{n_samples}_A{alpha}_with_refs.pkl", 'rb') as f:
                _, ref_vals = pickle.load(f)
            with open(f"{dirname}/data/Crit_vals_pwr/S{n_samples}_A{alpha}_with_refs.pkl", 'rb') as f:
                _, ref_vals_new = pickle.load(f)
        return ref_vals, ref_vals_new


    def _get_test_types(self):
        """Helper function for the poweR-based tests.

        Returns:
            dict: Dict of test functions from R.
        """
        testnames = ['kolmogorov',
            'CvM',
            'AD_pwr',
            'Durbin',
            'Kuiper',
            'HG1',
            'HG2',
            'Greenwood',
            'QM',
            'RC',
            'Moran',
            'Cressie1',
            'Cressie2',
            'Vasicek',
            'Swartz',
            'Morales',
            'Pardo',
            'Marhuenda',
            'Zhang1',
            'Zhang2']
        test_types_new = [pwr.create_alter(robjects.FloatVector(np.arange(63,83)))[i][0] for i in range(20)]
        return {k:v for k,v in zip(testnames, test_types_new)}

    def transform_to_reject_dt_corr(self, dt, alpha, n_samples, correction_method='fdr_bh'):
        """Apply p-value corrections on the dataframe of test statistics.

        Args:
            dt (dataframe): The DataFrame containing the calculated test statistics for each dimension.
            alpha (float): The threshold for statistical significance.
            n_samples (int): The sample size used for the statistical tests. Can only be 
                in [30,50,100,600]
            correction_method (str, optional): Which type of p-value correction to apply. Recommended is 'fdr_bh', 
                but 'fdr_by' and 'holm' are also supported.. Defaults to 'fdr_bh'.

        Returns:
            dataframe: Corrected test statistics.
        """
        reference_vals, ref_vals_new = self._load_ref_vals(n_samples)
        test_types_new = self._get_test_types()    
        
        dt_rejections = pd.DataFrame()
        dt_p_vals_temp = pd.DataFrame()
        for colname in self.p_value_columns:
            dt_rejections[colname] = multipletests(dt[colname], alpha=alpha, method=correction_method)[0]
            
        for k,v in reference_vals.items():
            if 'kurt' in k:
                temp = [percentileofscore(score=x, a=v, kind='mean')/100 for x in dt[k]]
                temp = [min(x, 1-x) for x in temp] #two-sided comparison
                dt_rejections[k] = multipletests(temp, alpha=alpha/2, method=correction_method)[0]
            elif k in ['min', 'wasserstein', 'mdd_max', 'mdd_min']:
                temp = [1-percentileofscore(score=x, a=v, kind='mean')/100 for x in dt[k]]
                dt_rejections[k] = multipletests(temp, alpha=alpha, method=correction_method)[0]
            else:
                temp = [percentileofscore(score=x, a=v, kind='mean')/100 for x in dt[k]]
                dt_rejections[k] = multipletests(temp, alpha=alpha, method=correction_method)[0]
        for k,v in ref_vals_new.items():
            if test_types_new[k] == 4:
                temp = [percentileofscore(score=x, a=v, kind='mean')/100 for x in dt[k]]
                dt_rejections[k] = multipletests(temp, alpha=alpha, method=correction_method)[0]
            else:
                temp = [1-percentileofscore(score=x, a=v, kind='mean')/100 for x in dt[k]]
                dt_rejections[k] = multipletests(temp, alpha=alpha, method=correction_method)[0]
        return dt_rejections

    def _get_test_names_dict(self):
        """Helper function to ensure consistent naming for the used statistical tests
        by creating a dictionary

        Returns:
            dict: Dict of all test functions.
        """
        test_dict_per = get_test_dict(n_samples=100, per_dim=True)
        test_names = list(test_dict_per.keys())
        test_names.remove('AD_pwr')
        test_names_paper = ['1-spacing',
            '2-spacing',
            '3-spacing',
            'range',
            'min',
            'max',
            'AD',
            'tAD',
            'Shapiro',
            'JB',
            'LD-min',
            'LD-max',
            'Kurt',
            'MPD-max',
            'MPD-min',
            'Wasserstein',
            'NS',
            'KS',
            'CvM',
            'Durbin',
            'Kuiper',
            'HG1',
            'HG2',
            'Greenwood',
            'QM',
            'RC',
            'Moran',
            'Cressie1',
            'Cressie2',
            'Vasicek',
            'Swartz',
            'Morales',
            'Pardo',
            'Marhuenda',
            'Zhang1',
            'Zhang2']

        test_label_dict = {k:v for k,v in zip(test_names, test_names_paper)}
        return test_label_dict


    def plot_swarm_with_heatmap(self, data, rejections, filename = None):
        """Plotting function to create the swarmplot and rejection heatmap.

        Args:
            data (dataframe): The DataFrame containing the final position values.
            rejections (dataframe): The DataFrame containing the corresponding test rejections.
            filename (string, optional): If not none, the name of the file to store the figure. Defaults to None.
        """
        test_label_dict = self._get_test_names_dict()
        data_dt = pd.DataFrame(data)
        fig, axs = plt.subplots(2, figsize=(19,14), sharex=True)
        ax1 = axs[0]
        dt_molt = data_dt.melt()
        dt_molt['variable'] = dt_molt['variable'] + 1.5
        sbs.swarmplot(data=dt_molt, x='variable', y='value', ax=ax1)
        ax1.set_xlim(-0.5, self.DIM-0.5)
        for dim in range(self.DIM):
            c0 = ax1.get_children()[dim]
            c0.set_offsets([[x+0.5,y] for x,y in c0.get_offsets()])
            ax1.axvline(dim, color='k', lw=0.6, ls=':')
        sbs.heatmap(np.array(rejections).transpose(), ax=axs[1], cbar=False, 
                    yticklabels=[test_label_dict[x] for x in rejections.columns], linewidths=.01, cmap='crest_r')

        ax1.set_xlabel("")
        axs[1].set_xlabel("Dimension", fontsize=16)
        axs[1].set_xticklabels(range(1,self.DIM+1), fontsize=14)
        axs[1].set_yticklabels(axs[1].get_yticklabels(), fontsize=14)
        ax1.set_ylabel("Value", fontsize=16)
        ax1.set_ylim(0,1)
        ax1.set_yticklabels([0,0.2,0.4,0.6,0.8,1], fontsize=14)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()
            
    
            
    def predict_type(self, dt_rej,  print_type = False):
        """Predict the type of bias using the rejection data.

        Args:
            dt_rej (dataframe): Dataframe containing rejection data.
            print_type (bool, optional): Whether to output the type to the standard output or not. Defaults to False.

        Returns:
            dict: Dict with the predicted Class and the Class_Probabilities
        """
        mean_rej = (np.mean(np.array(dt_rej), axis=0) >= 0.1)
        if np.sum(mean_rej) == 0:
            if print_type:
                print('No clear evidence of bias detected')
            return 'none'
        dirname = os.path.dirname(__file__)

        #download RF models if needed from 
        if (not os.path.isfile(f"{dirname}/models/RFs/rf_few_classes.pkl")):
            print("Downloading model files, this takes a while..")
            r = requests.get("https://figshare.com/ndownloader/files/31145971")
            zipfile = ZipFile(BytesIO(r.content))
            zipfile.extractall(f"{dirname}/models/")
        
        with open(f"{dirname}/models/RFs/rf_few_classes.pkl", "rb") as input_file:
            rf = pickle.load(input_file)
        res_class = rf.predict(mean_rej.reshape(1, -1))
        classes = rf.classes_
        prob_classes = rf.predict_proba(mean_rej.reshape(1, -1))
        
        with open(f"{dirname}/models/RFs/rf_scens.pkl", "rb") as input_file:
            rf = pickle.load(input_file)
        res_scen = rf.predict(mean_rej.reshape(1, -1))
        scennames = rf.classes_
        prob_scens = rf.predict_proba(mean_rej.reshape(1, -1))
        
        if print_type:
            print(f"Detected bias which seems to be related to {res_class} ({np.max(prob_classes):.2f} probability)." +
                f"The rejections seems to be most similar to the {res_scen} scenario ({np.max(prob_scens):.2f} probability).")
        return {'Class' : res_class[0], 'Class Probabilities' : prob_classes, 
                'Scenario' : res_scen[0], 'Scenario Probabilities' : prob_scens}

    def predict_deep(self, data, include_proba=True):
        """Predict the BIAS using our neural network.

        Args:
            data (dataframe): The matrix containing the final position values on F0. Note that these should be scaled 
                in [0,1], and in the shape (n_samples, dimension), where n_samples is in [50, 100] (later to add 30,600)
            include_proba (boolean, optional): To include the probabilities of each class or only the final label.
        
        Raises:
            ValueError: Unsupported sample size.

        Returns:
            predicted bias type (string), optional probabilities (array)
        """
        #load model
        n_samples = data.shape[1]
        if not n_samples in [50,100]:
            raise ValueError("Sample size is not supported")
        model = tf.keras.models.load_model(f"models/opt_cnn_model-{n_samples}.h5")
        targetnames = np.load("models/targetnames.npy")
        preds = model.predict([data])
        y = np.argmax(preds, axis=1)
        if include_proba:
            return targetnames[y], preds
        return targetnames[y]
        
    def predict(self, data, corr_method = 'fdr_bh', alpha=0.01, show_figure=False, filename = None, print_type = True):
        """The main function used to detect Structural Bias.

        Args:
            data (dataframe): The matrix containing the final position values on F0. Note that these should be scaled 
                in [0,1], and in the shape (n_samples, dimension), where n_samples is in [30, 50, 100, 600] 
            corr_method (str, optional): Which type of p-value correction to apply. Recommended is 'fdr_bh', 
                but 'fdr_by' and 'holm' are also supported.. Defaults to 'fdr_bh'.
            alpha (float, optional): The threshold for statistical significance. Defaults to 0.01.
            show_figure (bool, optional): Whether or not to create a plot of the final positions and the corresponding test rejections. Defaults to False.
            filename (string, optional): If not none, the name of the file to store the figure (only when show_figure is True). Defaults to None.
            print_type (bool, optional): Wheter or not to print the predicted type of SB. Defaults to True.

        Raises:
            ValueError: Unsupported sample size.

        Returns:
            dataframe, dict: rejection data, predicted Bias and type.
        """
        self.DIM = data.shape[1]
        n_samples = data.shape[0]
        if not n_samples in [30,50,100,600]:
            raise ValueError("Sample size is not supported")
        if print_type:
            print(f"Running SB calculation with {self.DIM}-dimensional data of sample size {n_samples} (alpha = {alpha})")
        records = {}    
        test_battery_per_dim = get_test_dict(n_samples)
        for tname, tfunc in test_battery_per_dim.items():
            temp = []
            for r in range(self.DIM):
                try:
                    temp.append(tfunc(data[:,r], alpha=alpha))
                except:
                    next
            records[tname] = temp
        dt = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in records.items() ]))
        dt_rejections = self.transform_to_reject_dt_corr(dt, alpha, n_samples, corr_method)
        #Drop duplicate test
        dt_rejections = dt_rejections.drop('AD_pwr', axis=1)
        
        if show_figure:
            self.plot_swarm_with_heatmap(data, dt_rejections, filename)
        
        return dt_rejections, self.predict_type(dt_rejections, print_type)


    def transform_to_reject_dt_across(self, dt, alpha, n_samples):
        """Transform rejection data for across dimension tests.

        Args:
            dt (dataframe): Rejection dataframe.
            alpha (float): Signficance level.
            n_samples (int): Number of samples.

        Returns:
            dataframe: Transformed rejection data.
        """
        crit_vals, crit_vals_new = self._load_ref_vals(n_samples, alpha, True)
        test_types_new = self._get_test_types()    
        
        dt_rejections = pd.DataFrame()
        for colname in self.p_value_columns:
            dt_rejections[colname] = dt[colname] < alpha
            
        #Ugly solution to distinguish two-sided vs one-sided tests
        dt_rejections['kurtosis'] = (crit_vals['kurtosis_low'] > dt['kurtosis']) | (dt['kurtosis'] > crit_vals['kurtosis_high'])
        dt_rejections['mmpd'] = (crit_vals['mmpd_low'] > dt['mmpd']) | (dt['mmpd'] > crit_vals['mmpd_high'])
        dt_rejections['mi'] = (crit_vals['mi_low'] > dt['mi']) | (dt['mi'] > crit_vals['mi_high'])
        dt_rejections['med_ddlud'] = (crit_vals['med_ddlud_low'] > dt['med_ddlud']) | (dt['med_ddlud'] > crit_vals['med_ddlud_high'])
        for k,v in crit_vals.items():
            if 'kurt' in k or 'low' in k or 'high' in k:
                next
            else:
                if k in ['max_ddlud']:
                    dt_rejections[k] = dt[k] > v
                else:
                    dt_rejections[k] = dt[k] < v
                    
        for k,v in crit_vals_new.items():
            if test_types_new[k] == 4:
                dt_rejections[k] = dt[k] < v
            else:
                dt_rejections[k] = dt[k] > v
        return dt_rejections

    def predict_multi_dim(self, data, alpha=0.01, print_type = True):
        """Predict Bias using across dimension tests.

        Args:
            data (dataframe): dataframe containing end positions.
            alpha (float, optional): Signficance level. Defaults to 0.01.
            print_type (bool, optional): Whether to output the type or not. Defaults to True.

        Raises:
            ValueError: unsupported sample size or dimension.

        Returns:
            list: List of failed tests that show potential bias.
        """
        DIM = data.shape[1]
        n_samples = data.shape[0]
        if not n_samples in [30,50,100,600]:
            raise ValueError("Sample size is not supported")
        if DIM != 30:
            raise ValueError("Only 30-dimensional data is supported for across-dimension testing")
        if print_type:
            print(f"Running SB calculation with {DIM}-dimensional data of sample size {n_samples} (alpha = {alpha})")
        records = {}    
        test_battery_across_dim = get_test_dict(n_samples, per_dim=False)
        for tname, tfunc in test_battery_across_dim.items():
            try:
                records[tname] = tfunc(data)
            except:
                next
        #TODO: fix this function
        dt = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in records.items() ]))
        dt_rejections = self.transform_to_reject_dt_across(dt, alpha, n_samples)
        failed_tests = [x for x in dt_rejections.columns if np.sum(dt_rejections[x]) > 0 ]
        if print_type:
            if len(failed_tests == 0):
                print('No clear evidence of bias detected')
            else:
                print(f'The following tests detected potential structural bias: {failed_tests}')
        return failed_tests