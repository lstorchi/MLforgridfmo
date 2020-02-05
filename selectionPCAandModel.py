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

        print("")
        print ("Starting with ", features_array.shape[0], \
                " observations and ", features_array.shape[1], " features ")


        print("")
        print("Standardize features...")
        features_array = StandardScaler().fit_transform(features_array)

        print ("")
        print ("Correlation vs label...")
        for k in featuresselected:
            i = featuresselected.index(k)
            P = scipy.stats.pearsonr(features_array[:, i], labels)[0]
            S = scipy.stats.spearmanr(features_array[:, i], labels)[0]
            if abs(P) > 0.5 or abs(S) > 0.5:
                print (" %30s "%k, " vs ", labelname, " P %6.3f S %6.3f "%(P, S))

        print ("")
        print ("Correlations remove over 0.9 ...")

        toremove = {}
        for k1 in featuresselected:
            i1 = featuresselected.index(k1)
            for k2 in featuresselected:
                if k1 != k2:
                    i2 = featuresselected.index(k2)
                    if i2 > i1:
                        P = scipy.stats.pearsonr(features_array[:, i1], features_array[:, i2])[0]
                        S = scipy.stats.spearmanr(features_array[:, i1], features_array[:, i2])[0]
                        if abs(P) > 0.9:
                            print (" %30s "%k1, " vs %30s "%k2 , " P %6.3f S %6.3f "%(P, S))
                            toremove[k1] = i1


        print ("")
        print ("Remove hoghly correlated features ...")
        features_array = np.delete(features_array, list(toremove.values()), axis=1)
        for k in toremove:
            print ("   remove %30s %3d "%(k, toremove[k]))       
            featuresselected.remove(k)

        #i = 0
        #for k in features:
        #    m = np.mean(features_array[:, i])
        #    d = np.std(features_array[:, i])
        #
        #    print("Label ", labelname, " has new mean: %10.5f"%m, " and STD: %10.5f"%d)
        #
        #    i = i + 1

        print("")
        print ("Using ", features_array.shape[0], \
                " observations and ", features_array.shape[1], " features ")

        pca = PCA()
        principalComponents = pca.fit_transform(features_array)

        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)') #for each component
        #plt.show()
        plt.savefig('pcavariance.pdf')

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

        pca = PCA(n_components=N)
        features = pca.fit_transform(features_array)

        num_of_run = 100

        for numoftrees in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 100]:

            mean_rms = 0.0
            mean_mae = 0.0
            mean_maxae = 0.0
            mean_rp = 0.0
            mean_score = 0.0
            
            mean_rms_train = 0.0
            mean_mae_train = 0.0
            mean_maxae_train = 0.0
            mean_rp_train = 0.0
            mean_score_train = 0.0

            print("Using %4d num. of tress"%numoftrees)
            for i in range(num_of_run):
            
                test = None
                train = None
                regressor = None
            
                #test, train, regressor = \
                #           util.lr_split_build_and_test (features, labels)
            
                test, train, regressor = \
                           util.rf_split_build_and_test (features_array, labels, 
                               numoftrees)
            
                mean_rms += test[0] 
                mean_mae += test[1]
                mean_maxae += test[2]
                mean_rp += test[3]
                mean_score += test[4]
               
                mean_rms_train += train[0] 
                mean_mae_train += train[1]
                mean_maxae_train += train[2]
                mean_rp_train += train[3]
                mean_score_train += train[4]

            print ("Mean  RMSE: %10.5f train: %10.5f "%((mean_rms/float(num_of_run)), \
                    (mean_rms_train/float(num_of_run)) ))
            print ("Mean   MAE: %10.5f train: %10.5f "%((mean_mae/float(num_of_run)), \
                    (mean_mae_train/float(num_of_run)) ))
            print ("Mean MaxAE: %10.5f train: %10.5f "%((mean_maxae/float(num_of_run)), \
                    (mean_maxae_train/float(num_of_run)) ))
            print ("Mean    rP: %10.5f train: %10.5f "%((mean_rp/float(num_of_run)), \
                    (mean_rp_train/float(num_of_run)) ))
            print ("Mean score: %10.5f train: %10.5f "%((mean_score/float(num_of_run)), \
                    (mean_score_train/float(num_of_run)) ))

