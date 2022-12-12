#imports
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import json
from tqdm import tqdm


from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
import autokeras as ak
#load data
from BIAS.SB_Test_runner import get_scens_per_dim, get_simulated_data
from BIAS import BIAS, f0, install_r_packages
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix



class biasmodel(MLPClassifier):
    def __init__(self, model, targetnames):
        self.model = model
        self.test = BIAS()
        self.targetnames = targetnames

    def predict(self, X):
        y = []
        for x in tqdm(X):
            rejec, pred = self.test.predict(x, show_figure=False, print_type=False)
            if (pred == "none"):
                class_pred = 0
            else:
                class_pred = 1
            y.append(class_pred)
            
        return np.array(y)



class deepbiasmodel(MLPClassifier):
    def __init__(self, model, targetnames):
        self.model = model
        self.test = BIAS()
        self.targetnames = targetnames

    def predict(self, X):
        y = []
        for x in tqdm(X):
            pred = self.test.predict_deep(x, include_proba=False)[0]
            if (pred == "unif"):
                class_pred = 0
            else:
                class_pred = 1
            y.append(class_pred)
            
        return np.array(y)

#settings for this experiment
rep = 2
for dim in [10,20,30,40]:
    for n_samples in [30,50,100,600]:#,
        #load data
        scenes = get_scens_per_dim()
        per_label = {"unif":0, "bias":0}
        X = []
        y = []
        realY = []
        for scene in scenes:
            label = scene[0]
            realLabel = f"{label} " + json.dumps(scene[1])
            kwargs = scene[1]
            if (label == "unif"):
                rep1 = 189 * rep
            else:
                rep1 = rep
            for r in range(rep1):
                data_arr = get_simulated_data(label, dim, n_samples, kwargs=kwargs)
                data = []
                for r in range(dim):
                    data.append(np.sort(data_arr[:,r]))
                X.append(np.array(data).T)
            if (label != "unif"):
                label = "bias"
            per_label[label] += rep1
            y.extend([label]*rep1)
        print(per_label)
        X = np.array(X)
        int_y, targetnames= pd.factorize(y)
        model1 = deepbiasmodel(MLPClassifier(), targetnames)
        pred1 = model1.predict(X)
        print(f"CLassification report for deep model dim {dim}, samples {n_samples}")
        report = classification_report(int_y, pred1, target_names=targetnames)
        with open(f'report_deep_{n_samples}-{dim}.txt', 'w') as f:
            f.write(report)
        print(report)
        fig, ax = plt.subplots(figsize=(14, 14))
        cm = confusion_matrix(int_y, pred1)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=targetnames)
        disp.plot(ax=ax) 
        plt.savefig(f"experiments/models/opt_cnn_model-{n_samples}-confusion_binary.png")

        
        model2 = biasmodel(MLPClassifier(), targetnames)
        pred2 = model2.predict(X)
        print(f"CLassification report for stat model dim {dim}, samples {n_samples}")
        report = classification_report(int_y, pred2, target_names=targetnames)
        with open(f'report_bias_{n_samples}-{dim}.txt', 'w') as f:
            f.write(report)
        print(report)
        fig, ax = plt.subplots(figsize=(14, 14))
        cm = confusion_matrix(int_y, pred2)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=targetnames)
        disp.plot(ax=ax)  
        plt.savefig(f"experiments/models/opt_bias_model-{n_samples}-confusion_binary.png")
        