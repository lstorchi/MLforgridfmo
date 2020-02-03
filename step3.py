from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.stats
import os.path
import sys

import math

import util

#####################################################################

if __name__== "__main__":
    
    filename = ""

    if len(sys.argv) != 2:
        print("usage: " , sys.argv[0], " file.xslx")
        exit(1)
    else:
        filename = sys.argv[1]

    labelname = "logKa"
    featuresselected = ["Etot",\
            "Pocket charge" , \
            "Pocket pos. char." , \
            "Pocket neg. char." , \
            "LogP", \
            "PvsL 5 A nomalized", \
            "LvsP 5 A nomalized", \
            "FvsF 5 A nomalized" , \
            "Ligand rotors", \
            "Ligand charge" \
            ]

    names = []
    features = {}
    labels = []

    if os.path.exists(filename):
        names, features, labels = util.readfile(filename, labelname)
    else:
        print("file ", filename , " does not exist ")

    if util.extract_featuresubset (features, featuresselected):

        print("All the features have been found")

        for k in features:
            m = np.mean(features[k])
            d = np.std(features[k])

            print(k, " has mean: ", m, " and STD: ", d)

        m = np.mean(labels)
        d = np.std(labels)
        print("Label ", labelname, " has mean: ", m, " and STD: ", d)

        #labels = labels - m
        #labels = labels / d

        features_array = np.zeros((len(labels), len(features.keys())))

        i = 0
        for k in features:
            features_array[:, i] = features[k]
            i = i + 1

        features_array = StandardScaler().fit_transform(features_array)

        i = 0
        for k in features:
            m = np.mean(features_array[:, i])
            d = np.std(features_array[:, i])

            print("Label ", labelname, " has new mean: ", m, " and STD: ", d)
 
            i = i + 1

