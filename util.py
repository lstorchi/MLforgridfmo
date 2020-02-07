import pandas as pd
import numpy as np

import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor  
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy.stats import pearsonr

LEAVEPERC = 0.15

#######################################################################

def _compute_results_parameters (regressor, 
        X_train, y_train, X_test, y_test):
      
    # fit the regressor with X and Y data 
    regressor.fit(X_train, y_train) 
    
    y_pred = regressor.predict(X_test) 
    
    rms = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_test, y_pred)]
    maxae = np.amax(absdiff)
    rp = pearsonr(y_test, y_pred)[0]
    score = regressor.score(X_test, y_test)
    r2 = r2_score(y_test, y_pred)

    y_pred = regressor.predict(X_train) 
    
    rmst = math.sqrt(mean_squared_error(y_train, y_pred))
    maet = mean_absolute_error(y_train, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_train, y_pred)]
    maxaet = np.amax(absdiff)
    rpt = pearsonr(y_train, y_pred)[0]
    scoret = regressor.score(X_train,y_train)
    r2t = r2_score(y_train, y_pred)

    return rms, mae, maxae, rp, score, r2, \
        rmst, maet, maxaet, rpt, scoret, r2t 

#######################################################################

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

#######################################################################

def difference(lst1, lst2):
    return list(set(lst1) - set(lst2))

#######################################################################

def readfile(filename, labelname, verbose = True):

    xls = pd.ExcelFile(filename)

    names = []
    features = {}
    labels = []

    thefirst = True
    for name in xls.sheet_names:
        if verbose:
            print ("Parsing data from ", name)
        df = xls.parse(name)
        ft = list(df.columns.values)
        if verbose:
            print("  reading names from ", ft[0])

        morethen = []
        if thefirst:
            names = list(df[ft[0]].values)
            thefirst = False
        else:
            localnames = list(df[ft[0]].values)
            if len(names) != len(localnames):

                morethen = difference(localnames, names)
                lessthen = difference(names, localnames)

                if len(morethen):
                    if verbose:
                        print("    WARNING has more entries ", morethen, " I will drop it")
                elif len(lessthen):
                    print("    ERROR has less entries ", lessthen)
                    exit(1)

        dfrn = df.set_index(ft[0], drop = True)

        if len(morethen) > 0:
            dfrn = dfrn.drop(morethen)

        for vn in dfrn.columns.values:
            if vn != labelname:
                features[vn] = dfrn.loc[:,vn].values
            else:
                labels =  dfrn.loc[:,vn].values


    # check values 
    dim = len(labels)

    if (dim != len(names)):
        print ("ERROR in dimensions")
        exit(1)

    for k in features:
        if dim != features[k].shape[0]:
            print ("ERROR in dimensions of ", k, " ", dim , " vs ", features[k].shape[0])
            exit(1)

    return names, features, labels

#######################################################################

def extract_featuresubset (features, featuresselected):

    somemissing = False
    for k in featuresselected:
        if k not in features:
            #print ("Error cannot find ", k)
            somemissing = True

    if not somemissing:
        #print("All the features have been found")

        toberemoved = []

        for allf in features:
            if allf not in featuresselected:
                toberemoved.append(allf)

        for ftorm in toberemoved:
            del features[ftorm]

        return True

    return False

#######################################################################

def rr_split_build_and_test (features_array, labels, \
        alpha = 0.1):

    X_train, X_test, y_train, y_test = train_test_split( \
            features_array, labels, test_size=LEAVEPERC)
    
    regressor = Ridge()

    rms, mae, maxae, rp, score, r2, rmst, maet, maxaet, rpt, scoret, r2t = \
            _compute_results_parameters (regressor, X_train, y_train, \
            X_test, y_test)
      
    return (rms, mae, maxae, rp, score, r2), \
            (rmst, maet, maxaet, rpt, scoret, r2t), \
            regressor

#####################################################################

def lr_split_build_and_test (features_array, labels):

    X_train, X_test, y_train, y_test = train_test_split( \
            features_array, labels, test_size=LEAVEPERC)
    
    regressor = LinearRegression()

    rms, mae, maxae, rp, score, r2, rmst, maet, maxaet, rpt, scoret, r2t = \
            _compute_results_parameters (regressor, X_train, y_train, \
            X_test, y_test)
      
    return (rms, mae, maxae, rp, score, r2), \
            (rmst, maet, maxaet, rpt, scoret, r2t), \
            regressor

#####################################################################

def rf_split_build_and_test (features_array, labels, \
        numofdecisiontree = 10):

    X_train, X_test, y_train, y_test = train_test_split( \
            features_array, labels, test_size=LEAVEPERC)
    
    regressor = RandomForestRegressor(n_estimators = numofdecisiontree, \
              random_state = 42)

    rms, mae, maxae, rp, score, r2, rmst, maet, maxaet, rpt, scoret, r2t = \
            _compute_results_parameters (regressor, X_train, y_train, \
            X_test, y_test)
      
    return (rms, mae, maxae, rp, score, r2), \
            (rmst, maet, maxaet, rpt, scoret, r2t), \
            regressor

#####################################################################

def rt_split_build_and_test (features_array, labels):

    X_train, X_test, y_train, y_test = train_test_split( \
            features_array, labels, test_size=LEAVEPERC)
    
    #regressor = DecisionTreeRegressor(random_state = 0)  
    regressor = DecisionTreeRegressor()

    rms, mae, maxae, rp, score, r2, rmst, maet, maxaet, rpt, scoret, r2t = \
            _compute_results_parameters (regressor, X_train, y_train, \
            X_test, y_test)
      
    return (rms, mae, maxae, rp, score, r2), \
            (rmst, maet, maxaet, rpt, scoret, r2t), \
            regressor

#####################################################################

def krr_split_build_and_test (features_array, labels, \
        alphain = 1.0, gammain = 1.0):

    X_train, X_test, y_train, y_test = train_test_split( \
            features_array, labels, test_size=LEAVEPERC)

    regressor = KernelRidge(alpha = alphain, kernel='rbf', gamma = gammain)
    #regressor = KernelRidge(kernel='linear', gamma=3.0e-4)
    #print(val_features.shape, val_labels.shape)

    rms, mae, maxae, rp, score, r2, rmst, maet, maxaet, rpt, scoret, r2t = \
            _compute_results_parameters (regressor, X_train, y_train, \
            X_test, y_test)
      
    return (rms, mae, maxae, rp, score, r2), \
            (rmst, maet, maxaet, rpt, scoret, r2t), \
            regressor

#####################################################################

def krr_param_selection(x, y, nfolds):
    
    alphas = [100, 10, 1e0, 0.1, 1e-2, 1e-3, 3.0e-4, 1e-4]
    gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                     
    param_grid = {'alpha': alphas, 'gamma' : gammas}
    grid_search = GridSearchCV(KernelRidge(kernel='rbf'), \
    param_grid, n_jobs = 5, cv=nfolds)
    grid_search.fit(x, y)
                                                                 
    return grid_search

#####################################################################
