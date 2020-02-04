import pandas as pd
import numpy as np

import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

from scipy.stats import pearsonr

LEAVEPERC = 0.15

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
      
    # fit the regressor with X and Y data 
    regressor.fit(X_train, y_train) 
    
    y_pred = regressor.predict(X_test) 
    
    rms = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_test, y_pred)]
    maxae = np.amax(absdiff)
    rp = pearsonr(y_test, y_pred)[0]

    y_pred = regressor.predict(X_train) 
    
    rmstrain = math.sqrt(mean_squared_error(y_train, y_pred))
    maetrain = mean_absolute_error(y_train, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_train, y_pred)]
    maxaetrain = np.amax(absdiff)
    rptrain = pearsonr(y_train, y_pred)[0]

    train_score = regressor.score(X_train,y_train)
    score = regressor.score(X_test, y_test)

    return (rms, mae, maxae, rp, score), \
            (rmstrain, maetrain, maxaetrain, rptrain, train_score), \
            regressor

#####################################################################

def lr_split_build_and_test (features_array, labels):

    X_train, X_test, y_train, y_test = train_test_split( \
            features_array, labels, test_size=LEAVEPERC)
    
    regressor = LinearRegression()
      
    # fit the regressor with X and Y data 
    regressor.fit(X_train, y_train) 
    
    y_pred = regressor.predict(X_test) 
    
    rms = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_test, y_pred)]
    maxae = np.amax(absdiff)
    rp = pearsonr(y_test, y_pred)[0]

    y_pred = regressor.predict(X_train) 
    
    rmstrain = math.sqrt(mean_squared_error(y_train, y_pred))
    maetrain = mean_absolute_error(y_train, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_train, y_pred)]
    maxaetrain = np.amax(absdiff)
    rptrain = pearsonr(y_train, y_pred)[0]

    train_score = regressor.score(X_train,y_train)
    score = regressor.score(X_test, y_test)

    return (rms, mae, maxae, rp, score), \
            (rmstrain, maetrain, maxaetrain, rptrain, train_score), \
            regressor

#####################################################################

def rf_split_build_and_test (features_array, labels, \
        numofdecisiontree = 10):

    X_train, X_test, y_train, y_test = train_test_split( \
            features_array, labels, test_size=LEAVEPERC)
    
    regressor = RandomForestRegressor(n_estimators = numofdecisiontree, \
              random_state = 42)
      
    # fit the regressor with X and Y data 
    regressor.fit(X_train, y_train) 
    
    y_pred = regressor.predict(X_test) 
    
    rms = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_test, y_pred)]
    maxae = np.amax(absdiff)
    rp = pearsonr(y_test, y_pred)[0]

    y_pred = regressor.predict(X_train) 
    
    rmstrain = math.sqrt(mean_squared_error(y_train, y_pred))
    maetrain = mean_absolute_error(y_train, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_train, y_pred)]
    maxaetrain = np.amax(absdiff)
    rptrain = pearsonr(y_train, y_pred)[0]

    train_score = regressor.score(X_train,y_train)
    score = regressor.score(X_test, y_test)

    return (rms, mae, maxae, rp, score), \
            (rmstrain, maetrain, maxaetrain, rptrain, train_score), \
            regressor

#####################################################################

def rt_split_build_and_test (features_array, labels):

    X_train, X_test, y_train, y_test = train_test_split( \
            features_array, labels, test_size=LEAVEPERC)
    
    #regressor = DecisionTreeRegressor(random_state = 0)  
    regressor = DecisionTreeRegressor()
      
    # fit the regressor with X and Y data 
    regressor.fit(X_train, y_train) 
    
    y_pred = regressor.predict(X_test) 
    
    rms = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_test, y_pred)]
    maxae = np.amax(absdiff)
    rp = pearsonr(y_test, y_pred)[0]

    y_pred = regressor.predict(X_train) 
    
    rmstrain = math.sqrt(mean_squared_error(y_train, y_pred))
    maetrain = mean_absolute_error(y_train, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_train, y_pred)]
    maxaetrain = np.amax(absdiff)
    rptrain = pearsonr(y_train, y_pred)[0]

    train_score = regressor.score(X_train,y_train)
    score = regressor.score(X_test, y_test)

    return (rms, mae, maxae, rp, score), \
            (rmstrain, maetrain, maxaetrain, rptrain, train_score), \
            regressor

#####################################################################

def krr_split_build_and_test (features_array, labels, \
        alphain = 1.0, gammain = 1.0):

    X_train, X_test, y_train, y_test = train_test_split( \
            features_array, labels, test_size=LEAVEPERC)

    clf = KernelRidge(alpha = alphain, kernel='rbf', gamma = gammain)
    #clf = KernelRidge(kernel='linear', gamma=3.0e-4)
    #print(val_features.shape, val_labels.shape)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    #regressor = DecisionTreeRegressor(random_state = 0)  
    regressor = DecisionTreeRegressor()
      
    # fit the regressor with X and Y data 
    regressor.fit(X_train, y_train) 
    
    y_pred = regressor.predict(X_test) 
    
    rms = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_test, y_pred)]
    maxae = np.amax(absdiff)
    rp = pearsonr(y_test, y_pred)[0]

    y_pred = regressor.predict(X_train) 
    
    rmstrain = math.sqrt(mean_squared_error(y_train, y_pred))
    maetrain = mean_absolute_error(y_train, y_pred)
    absdiff = [abs(x - y) for x, y in zip(y_train, y_pred)]
    maxaetrain = np.amax(absdiff)
    rptrain = pearsonr(y_train, y_pred)[0]

    train_score = regressor.score(X_train,y_train)
    score = regressor.score(X_test, y_test)

    return (rms, mae, maxae, rp, score), \
            (rmstrain, maetrain, maxaetrain, rptrain, train_score), \
            regressor

#####################################################################


