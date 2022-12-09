#imports
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import json
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
import autokeras as ak
#load data
from BIAS.SB_Test_runner import get_scens_per_dim, get_simulated_data
from BIAS import BIAS, f0, install_r_packages

class newmodel(MLPClassifier):
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        y = self.model.predict(X)
        return np.argmax(y, axis=1)

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
                class_pred = "unif"
            elif (pred["Class"] == "Bounds"):
                class_pred = "bounds"
            elif (pred["Class"] == "Gaps"):
                class_pred = "gaps/clusters"
            elif (pred["Class"] == "Clusters"):
                class_pred = "gaps/clusters"
            elif (pred["Class"] == "Center"):
                class_pred = "centre"
            elif (pred["Class"] == "Discretization"):
                class_pred = "disc"
            else:
                print(pred)
            y.append(np.where(targetnames == class_pred)[0])
            
        return np.array(y)

#settings for this experiment
rep = 20000

for n_samples in [600]:#,50,100,600
    #load data
    scenes = get_scens_per_dim()
    per_label = {"unif":0, "centre":0, "bounds":0, "gaps/clusters":0, "disc":0}
    X = []
    y = []
    realY = []
    for scene in scenes:
        label = scene[0]
        realLabel = f"{label} " + json.dumps(scene[1])
        kwargs = scene[1]
        if (label == "unif"):
            rep1 = 4 * rep
        elif (label in ["trunc_unif", "cauchy", "norm"]):
            rep1 = int(rep / 32)
        elif (label in ["bound_thing","inv_norm", "inv_cauchy"]):
            rep1 = int(rep / 48)
        elif (label in ["clusters","gaps", "part_unif"]):
            rep1 = int(rep / 67)
        elif (label in ["spikes", "shifted_spikes"]):
            rep1 = int(rep / 42)
        data = get_simulated_data(label, rep=rep1, n_samples = n_samples, kwargs=kwargs)
        for r in range(rep1):
            X.append(np.sort(data[:,r]))
        if (label in ["trunc_unif", "cauchy", "norm"]):
            label = "centre"
        elif (label in ["bound_thing","inv_norm", "inv_cauchy"]):
            label = "bounds"
        elif (label in ["gaps", "part_unif", "clusters"]):
            label = "gaps/clusters"
        elif (label in ["spikes", "shifted_spikes"]):
            label = "disc"
        per_label[label] += rep1
        y.extend([label]*rep1)
        realY.extend([realLabel] * rep1)

    X = np.array(X)
    int_y, targetnames= pd.factorize(y)
    int_real_y, targetnames_real= pd.factorize(realY)

    cat_y = to_categorical(int_y)
    cat_y_real = to_categorical(int_real_y)

    X_train, X_test, y_train, y_test = train_test_split(X, cat_y, test_size=0.2, random_state=42, stratify=int_y)
    _, __, y_train_real, y_test_real = train_test_split(X, cat_y_real, test_size=0.2, random_state=42, stratify=int_y)

    #expand dims
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    #load model
    from sklearn.metrics import f1_score

    model = tf.keras.models.load_model(f"BIAS/models/opt_cnn_model-{n_samples}.h5")
    #model.save(f"BIAS/models/opt_cnn_model-{n_samples}.tf")
    #model.summary()
    print(
        "Accuracy: {accuracy}".format(
            accuracy = model.evaluate(x=X_test, y=y_test)
        ),
        "f1 score: {f1}".format(
            f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1), average='macro')
        )
    )
    tf.keras.utils.plot_model(model, to_file=f"experiments/models/opt_cnn_model-{n_samples}.png")

    model1 = newmodel(model)
    hat_y_real = model.predict(X_test)
    hat_y = np.argmax(hat_y_real, axis=1)

    test_y = np.argmax(y_test, axis=1)
    test_real_y = np.argmax(y_test_real, axis=1)
    fig, ax = plt.subplots(figsize=(14, 14))
    #np.save("targetnames.npy", targetnames)
    plot_confusion_matrix(model1, X_test, test_y, normalize='true', xticks_rotation = 'vertical', display_labels = targetnames, ax=ax) 
    plt.savefig(f"experiments/models/opt_cnn_model-{n_samples}-confusion.png")

    #analyse misclassifications
    misclassifications_per_scenario = {}
    for i in range(len(hat_y)):
        if hat_y[i] != test_y[i]:
            if (targetnames_real[test_real_y[i]] not in misclassifications_per_scenario.keys()):
                misclassifications_per_scenario[targetnames_real[test_real_y[i]]] = 1
            misclassifications_per_scenario[targetnames_real[test_real_y[i]]] += 1
    #print(misclassifications_per_scenario)
    # Serializing json
    json_object = json.dumps(misclassifications_per_scenario, indent=4)
    
    # Writing to sample.json
    with open(f"experiments/misclassifications_per_scenario_{n_samples}.json", "w") as outfile:
        outfile.write(json_object)

    #compare with classifical method
    #do 30 independent runs (5 dimensions)
    
    if False:

        model2 = biasmodel(model, targetnames)
        test_y = np.argmax(y_test, axis=1)
        #print(f1_score(np.argmax(y_test, axis=1), np.argmax(model2.predict(X_test), axis=1), average='macro')
        fig, ax = plt.subplots(figsize=(14, 14))
        plot_confusion_matrix(model2, X_test, test_y, normalize='true', xticks_rotation = 'vertical', display_labels = targetnames, ax=ax) 
        plt.savefig(f"experiments/models/bias_model-{n_samples}-confusion.png")