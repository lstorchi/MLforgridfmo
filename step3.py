from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.stats
import os.path
import sys

import math

import util

#####################################################################

def rf_split_build_and_test (features_array, labels):

     X_train, X_test, y_train, y_test = train_test_split( \
             features_array, labels, test_size=0.10)
     
     regressor = DecisionTreeRegressor(random_state = 0)  
       
     # fit the regressor with X and Y data 
     regressor.fit(X_train, y_train) 
     
     y_pred = regressor.predict(X_test) 
     
     #for i in range(y_pred.shape[0]):
     #    print(y_pred[i], " ", y_test[i])
     
     rms = math.sqrt(mean_squared_error(y_test, y_pred))
     mae = mean_absolute_error(y_test, y_pred)
     absdiff = [abs(x - y) for x, y in zip(y_test, y_pred)]
     maxae = np.amax(absdiff)

     rp = pearsonr(y_test, y_pred)[0]

     return rms, mae, maxae, rp, y_test, y_pred

#####################################################################

if __name__== "__main__":
    
    filename = ""

    if len(sys.argv) != 2:
        print("usage: " , sys.argv[0] , " file.xslx")
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
            "Ligand charge"]

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
            features[k] = features[k] - m
            features[k] = features[k] / d

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

        #split tran and test set

        mean_rms = 0.0
        mean_mae = 0.0
        mean_maxae = 0.0
        mean_rp = 0.0
        num_of_run = 200

        for i in range(num_of_run):

           rms, mae, maxae, rp, y_pred, y_test = \
                   rf_split_build_and_test (features_array, labels)
           
           #print (" RMSE: ", rms)
           #print ("  MAE: ", mae)
           #print ("MaxAE: ", maxae)

           mean_rms += rms
           mean_mae += mae
           mean_maxae += maxae
           mean_rp += rp

        print ("Mean  RMSE: ", mean_rms/float(num_of_run))
        print ("Mean   MAE: ", mean_mae/float(num_of_run))
        print ("Mean MaxAE: ", mean_maxae/float(num_of_run))
        print ("Mean    rP: ", mean_rp/float(num_of_run))

