
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import chi2
from sklearn.linear_model import Lasso


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.stats
import os.path
import sys

import math

import util

#####################################################################

def krr_param_selection(x, y, nfolds):
    
    alphas = [0.00001, 0.001, 0.01, 0.1, 1, 10]
    gammas = [0.00001, 0.001, 0.01, 0.1, 1, 10]

    alphas = [1e0, 0.1, 1e-2, 1e-3, 3.0e-4, 1e-4]
    gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                     
    param_grid = {'alpha': alphas, 'gamma' : gammas}
    grid_search = GridSearchCV(KernelRidge(kernel='rbf'), \
    param_grid, n_jobs = 5, cv=nfolds)
    grid_search.fit(x, y)
                                                                 
    return grid_search

#####################################################################

if __name__== "__main__":

    util.LEAVEPERC = 0.15
    
    filename = ""
    method = 0

    if len(sys.argv) != 3:
        print("usage: " , sys.argv[0] , \
                " file.xslx method=[1=rtree,2=krr,3=rforest,4=LinearR,5=Ridge]")
        exit(1)
    else:
        filename = sys.argv[1]
        method = int(sys.argv[2])

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

        #split tran and test set

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

        num_of_run = 200

        grid = (krr_param_selection (features_array, labels, 5))
        params = grid.best_params_
        print ("Best score: ", grid.best_score_)
        print ("Optimized params: ", params)

        for i in range(num_of_run):

            test = None
            train = None
            regressor = None

            if method == 1:
                test, train, regressor = \
                    util.rt_split_build_and_test (features_array, labels)
            elif method == 2:
                test, train, regressor = \
                    util.krr_split_build_and_test (features_array, labels, \
                    params["alpha"], params["gamma"])
            elif method == 3:
                test, train, regressor = \
                        util.rf_split_build_and_test (features_array, labels, 
                           20)

                print(regressor.feature_importances_)
            elif method == 4:
                test, train, regressor = \
                        util.lr_split_build_and_test (features_array, labels)
            elif method == 5:
                test, train, regressor = \
                        util.rr_split_build_and_test (features_array, labels)

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

        print ("Mean  RMSE: %10.5f"%(mean_rms/float(num_of_run)) )
        print ("Mean   MAE: %10.5f"%(mean_mae/float(num_of_run)) )
        print ("Mean MaxAE: %10.5f"%(mean_maxae/float(num_of_run)) )
        print ("Mean    rP: %10.5f"%(mean_rp/float(num_of_run)) )
        print ("Mean score: %10.5f"%(mean_score/float(num_of_run)) )

        print ("Mean  RMSE train: %10.5f"%(mean_rms_train/float(num_of_run)) )
        print ("Mean   MAE train: %10.5f"%(mean_mae_train/float(num_of_run)) )
        print ("Mean MaxAE train: %10.5f"%(mean_maxae_train/float(num_of_run)) )
        print ("Mean    rP train: %10.5f"%(mean_rp_train/float(num_of_run)) )
        print ("Mean score train: %10.5f"%(mean_score_train/float(num_of_run)) )

        X_train, X_test, y_train, y_test = train_test_split( \
             features_array, labels, test_size=util.LEAVEPERC)
 
        for a in [0.001, 0.01, 0.1, 1.0]:
            regressor = Lasso(alpha=a, max_iter=10e5)
            regressor.fit(X_train,y_train)

            train_score = regressor.score(X_train,y_train)
            test_score = regressor.score(X_test,y_test)
            coeff_used = np.sum(regressor.coef_!=0)

            print ("Lasso using alpha %10.5f "%(a))
            print ("  score train %10.5f "%(train_score)) 
            print ("  score test  %10.5f "%(test_score))
            print ("  number of features used ", coeff_used)
            for cidx in range(len(regressor.coef_)):
                if regressor.coef_[cidx] != 0.0:
                    print ("   ", cidx+1, " => ", featuresselected[cidx] )

