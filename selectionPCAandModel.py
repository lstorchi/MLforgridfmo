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
import argparse
import os.path
import math
import util
import sys

#####################################################################

def buildmdl_and_test (features_array, labels, method, num_of_run):

    if method == 1:

        for numoftrees in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
       
            mean_rms = 0.0
            mean_mae = 0.0
            mean_maxae = 0.0
            mean_rp = 0.0
            mean_score = 0.0
            mean_r2 = 0.0
            
            mean_rms_train = 0.0
            mean_mae_train = 0.0
            mean_maxae_train = 0.0
            mean_rp_train = 0.0
            mean_score_train = 0.0
            mean_r2_train = 0.0
       
            print("Using %4d num. of tress"%numoftrees)

            for i in range(num_of_run):
            
                test = None
                train = None
                regressor = None
            
                test, train, regressor = \
                           util.rf_split_build_and_test (features_array, labels, 
                               numoftrees)
            
                mean_rms += test[0] 
                mean_mae += test[1]
                mean_maxae += test[2]
                mean_rp += test[3]
                mean_score += test[4]
                mean_r2 += test[5]
       
                mean_rms_train += train[0] 
                mean_mae_train += train[1]
                mean_maxae_train += train[2]
                mean_rp_train += train[3]
                mean_score_train += train[4]
                mean_r2_train += train[4]
       
            print ("mean  rmse: %10.5f train: %10.5f "%((mean_rms/float(num_of_run)), \
                    (mean_rms_train/float(num_of_run)) ))
            print ("mean   mae: %10.5f train: %10.5f "%((mean_mae/float(num_of_run)), \
                    (mean_mae_train/float(num_of_run)) ))
            print ("mean maxae: %10.5f train: %10.5f "%((mean_maxae/float(num_of_run)), \
                    (mean_maxae_train/float(num_of_run)) ))
            print ("mean    rp: %10.5f train: %10.5f "%((mean_rp/float(num_of_run)), \
                    (mean_rp_train/float(num_of_run)) ))
            print ("mean score: %10.5f train: %10.5f "%((mean_score/float(num_of_run)), \
                    (mean_score_train/float(num_of_run)) ))
            print ("mean    r2: %10.5f train: %10.5f "%((mean_r2/float(num_of_run)), \
                    (mean_r2_train/float(num_of_run)) ))

    elif method == 2:
        
        grid = util.krr_param_selection (features_array, labels, 5)
        params = grid.best_params_
        print ("Best score: ", grid.best_score_)
        print ("Optimized params: ", params)


        mean_rms = 0.0
        mean_mae = 0.0
        mean_maxae = 0.0
        mean_rp = 0.0
        mean_score = 0.0
        mean_r2 = 0.0
        
        mean_rms_train = 0.0
        mean_mae_train = 0.0
        mean_maxae_train = 0.0
        mean_rp_train = 0.0
        mean_score_train = 0.0
        mean_r2_train = 0.0
        
        for i in range(num_of_run):
        
            test = None
            train = None
            regressor = None
        
            test, train, regressor = \
                    util.krr_split_build_and_test (features_array, labels, \
                    params["alpha"], params["gamma"])
        
        
            mean_rms += test[0] 
            mean_mae += test[1]
            mean_maxae += test[2]
            mean_rp += test[3]
            mean_score += test[4]
            mean_r2 += test[5]
        
            mean_rms_train += train[0] 
            mean_mae_train += train[1]
            mean_maxae_train += train[2]
            mean_rp_train += train[3]
            mean_score_train += train[4]
            mean_r2_train += train[4]
        
        print ("mean  rmse: %10.5f train: %10.5f "%((mean_rms/float(num_of_run)), \
                (mean_rms_train/float(num_of_run)) ))
        print ("mean   mae: %10.5f train: %10.5f "%((mean_mae/float(num_of_run)), \
                (mean_mae_train/float(num_of_run)) ))
        print ("mean maxae: %10.5f train: %10.5f "%((mean_maxae/float(num_of_run)), \
                (mean_maxae_train/float(num_of_run)) ))
        print ("mean    rp: %10.5f train: %10.5f "%((mean_rp/float(num_of_run)), \
                (mean_rp_train/float(num_of_run)) ))
        print ("mean score: %10.5f train: %10.5f "%((mean_score/float(num_of_run)), \
                (mean_score_train/float(num_of_run)) ))
        print ("mean    r2: %10.5f train: %10.5f "%((mean_r2/float(num_of_run)), \
                (mean_r2_train/float(num_of_run)) ))

    elif method == 3:
        
        mean_rms = 0.0
        mean_mae = 0.0
        mean_maxae = 0.0
        mean_rp = 0.0
        mean_score = 0.0
        mean_r2 = 0.0
        
        mean_rms_train = 0.0
        mean_mae_train = 0.0
        mean_maxae_train = 0.0
        mean_rp_train = 0.0
        mean_score_train = 0.0
        mean_r2_train = 0.0
        
        for i in range(num_of_run):
        
            test = None
            train = None
            regressor = None
        
            test, train, regressor = \
                    util.lr_split_build_and_test (features_array, labels)
        
            mean_rms += test[0] 
            mean_mae += test[1]
            mean_maxae += test[2]
            mean_rp += test[3]
            mean_score += test[4]
            mean_r2 += test[5]
        
            mean_rms_train += train[0] 
            mean_mae_train += train[1]
            mean_maxae_train += train[2]
            mean_rp_train += train[3]
            mean_score_train += train[4]
            mean_r2_train += train[4]
        
        print ("mean  rmse: %10.5f train: %10.5f "%((mean_rms/float(num_of_run)), \
                (mean_rms_train/float(num_of_run)) ))
        print ("mean   mae: %10.5f train: %10.5f "%((mean_mae/float(num_of_run)), \
                (mean_mae_train/float(num_of_run)) ))
        print ("mean maxae: %10.5f train: %10.5f "%((mean_maxae/float(num_of_run)), \
                (mean_maxae_train/float(num_of_run)) ))
        print ("mean    rp: %10.5f train: %10.5f "%((mean_rp/float(num_of_run)), \
                (mean_rp_train/float(num_of_run)) ))
        print ("mean score: %10.5f train: %10.5f "%((mean_score/float(num_of_run)), \
                (mean_score_train/float(num_of_run)) ))
        print ("mean    r2: %10.5f train: %10.5f "%((mean_r2/float(num_of_run)), \
                (mean_r2_train/float(num_of_run)) ))


#####################################################################

if __name__== "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-f","--filename", help="Excel filename", \
            type=str, required=True, dest="filename")
    parser.add_argument("-m", "--method", help="Method to use (default 1) 1=RandomForest 2=KernelRidgeRegression " + \
            " 3=LinearRegression", default=1, type=int)
    parser.add_argument("-n", "--num-of-iterations", help="Specify how many l10-out iterations dafault=100", \
            default=100, type=int, dest="numofiteration")
    parser.add_argument("-l", "--label", help="Specify the labelname default=logKa", \
            default="logKa", type=str)
    parser.add_argument("-p", "--leave-perc", help="Specify the tran/test split default=0.15", \
            default=0.15, type=float, dest="leaveperc")
    parser.add_argument("-F", "--features", help="Specify the fautures list default=all " +
            "otherwise specify the list first;second;...;last", \
            default="all", type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

    args = parser.parse_args()
    filename = args.filename
    method = args.method
    num_of_run = args.numofiteration
    labelname = args.label
    util.LEAVEPERC = args.leaveperc

    features_to_select = []

    #features_to_select = ["Etot",\
    #        "Pocket charge" , \
    #        "Pocket pos. char." , \
    #        "Pocket neg. char." , \
    #        "LogP", \
    #        "PvsL 5 A nomalized", \
    #        "LvsP 5 A nomalized", \
    #        "FvsF 5 A nomalized" , \
    #        "Ligand rotors", \
    #        "Ligand charge" \
    #        ]

    names = []
    features = {}
    labels = []

    if os.path.exists(filename):
        names, features, labels = util.readfile(filename, labelname, False)
    else:
        print("file ", filename , " does not exist ")

    if len(args.features.split(";")) == 1:
        if (args.features == "all"):
            features_to_select.clear()
            for k in features.keys():
                features_to_select.append(k)
    else:
        features_to_select = args.features.split(";")

    if util.extract_featuresubset (features, features_to_select):

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


        print ("")
        print ("Correlation vs label...")
        for k in features.keys():
            i = list(features.keys()).index(k)
            P = scipy.stats.pearsonr(features_array[:, i], labels)[0]
            S = scipy.stats.spearmanr(features_array[:, i], labels)[0]
            if abs(P) > 0.5 or abs(S) > 0.5:
                print (" %30s "%k, " vs ", labelname, " P %6.3f S %6.3f "%(P, S))

        print ("")
        print ("Correlations remove over 0.9 ...")

        toremove = {}
        for k1 in features:
            i1 = list(features.keys()).index(k1)
            for k2 in features:
                if k1 != k2:
                    i2 = list(features.keys()).index(k2)
                    if i2 > i1:
                        P = scipy.stats.pearsonr(features_array[:, i1], features_array[:, i2])[0]
                        S = scipy.stats.spearmanr(features_array[:, i1], features_array[:, i2])[0]
                        if abs(P) > 0.9:
                            print (" %30s "%k1, " vs %30s "%k2 , " P %6.3f S %6.3f "%(P, S))
                            toremove[k1] = i1


        print ("")
        print ("Remove highly correlated features ...")
        features_array = np.delete(features_array, list(toremove.values()), axis=1)
        for k in toremove:
            print ("   remove %30s %3d "%(k, toremove[k]))       
            del features[k]
            features_to_select.remove(k)

        print ("")
        print ("Chek problem in lits ...")
        for k in features:
            i = list(features.keys()).index(k)
            m = np.mean(features_array[:, i])
            d = np.std(features_array[:, i])

            m1 = np.mean(features[k])
            d1 = np.std(features[k])

            if k != features_to_select[i] or \
                    m != m1 or d != d1:
                print("Error list is not congruent")
                exit(-1)

        print("")
        print("Normalize features...")
        features_array = StandardScaler().fit_transform(features_array)

        for k in features:
            i = list(features.keys()).index(k)

            m = np.mean(features_array[:, i])
            d = np.std(features_array[:, i])

            print("%30s"%k, " has mean: %10.5f"%m, " and STD: %10.5f"%d)
        
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
        selectedvariables_peso = {}
        for v in pca.explained_variance_ratio_ :
            print("     PC%3d %10.5f"%(i, v*100.0))
            arr = np.array(abs(pca.components_[i-1]))
            for f in arr.argsort()[-N:][::-1]:
                print("       %3d ==> [%10.5f] %30s"%(f, \
                        arr[f], list(features.keys())[f]))

                if list(features.keys())[f] in selectedvariables:
                    selectedvariables[list(features.keys())[f]] += 1
                    selectedvariables_peso[list(features.keys())[f]] += arr[f]*v
                else:
                    selectedvariables[list(features.keys())[f]] = 1
                    selectedvariables_peso[list(features.keys())[f]] = arr[f]*v

            i = i + 1

        print("")
        print("Most important variables are: ")
        for v in selectedvariables.keys():
            print("  %30s ==> %5d %10.5f"%(v, selectedvariables[v], \
                    selectedvariables_peso[v]))
        print("")

        pca = PCA(n_components=N)
        features_array_PC = pca.fit_transform(features_array)

        print("Check using PC...")
        buildmdl_and_test (features_array_PC, labels, method, num_of_run)

        print("Check using features...")
        buildmdl_and_test (features_array, labels, method, num_of_run)

