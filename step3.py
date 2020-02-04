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
from sklearn.decomposition import PCA

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
        names, features, labels = util.readfile(filename, labelname, False)
    else:
        print("file ", filename , " does not exist ")


    featuresselected.clear()
    for k in features.keys():
        featuresselected.append(k)

    if util.extract_featuresubset (features, featuresselected):

        print("All the features have been found")

        for k in features:
            m = np.mean(features[k])
            d = np.std(features[k])

            print("%30s"%k, " has mean: %10.5f"%m, " and STD: %10.5f"%d)

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

        print("")
        print("Standardize features...")
        i = 0
        for k in features:
            m = np.mean(features_array[:, i])
            d = np.std(features_array[:, i])

            #print("Label ", labelname, " has new mean: %10.5f"%m, " and STD: %10.5f"%d)
 
            i = i + 1

        print("")
        print ("Features selection using ", features_array.shape[0], \
                " observations and ", features_array.shape[1], " features ")

        #pca = PCA()
        #principalComponents = pca.fit_transform(features_array)

        #plt.figure()
        #plt.plot(np.cumsum(pca.explained_variance_ratio_))
        #plt.xlabel('Number of Components')
        #plt.ylabel('Variance (%)') #for each component
        #plt.show()

        pca = PCA(0.95)
        principalComponents = pca.fit_transform(features_array)

        print("Number of components: ", pca.n_components_)
        print("  Explained variance: ")
        i = 1
        N = pca.n_components_
        selectedvariables = {}

        for v in pca.explained_variance_ratio_ :
            print("     PC%3d %10.5f"%(i, v*100.0))
            arr = np.array(abs(pca.components_[i-1]))
            for f in arr.argsort()[-N:][::-1]:
                print("       %3d ==> [%10.5f] %30s"%(f, \
                        arr[f], featuresselected[f]))

                if featuresselected[f] in selectedvariables:
                    selectedvariables[featuresselected[f]] += 1
                else:
                    selectedvariables[featuresselected[f]] = 1

            i = i + 1

        print("")
        print("Most important variables are: ")
        for v in selectedvariables.keys():
            print("  %30s ==> %5d"%(v, selectedvariables[v]))
        print("")

