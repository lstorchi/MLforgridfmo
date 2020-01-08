import pandas as pd
import numpy as nm

import scipy.stats
import os.path
import sys


#######################################################################

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

#######################################################################

def difference(lst1, lst2):
    return list(set(lst1) - set(lst2))

#######################################################################

def readfile(filename, labelname):

    xls = pd.ExcelFile(filename)

    names = []
    features = {}
    labels = []

    thefirst = True
    for name in xls.sheet_names:
        print ("Parsing data from ", name)
        df = xls.parse(name)
        ft = list(df.columns.values)
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

if __name__== "__main__":
    
    filename = ""

    if len(sys.argv) != 2:
        print("usage: " , sys.argv[0] , " file.xslx")
        exit(1)
    else:
        filename = sys.argv[1]

    labelname = "logKa"

    names = []
    features = {}
    labels = []

    if os.path.exists(filename):
        names, features, labels = readfile(filename, labelname)
    else:
        print("file ", filename , " does not exist ")

    for k in features:

        m = nm.mean(features[k])
        d = nm.std(features[k])

        #print(k, " has mean: ", m, " and STD: ", d)

        features[k] = features[k] - m
        features[k] = features[k] / d

    for k1 in features:
        for k2 in features:
            if k1 != k2:
                P = scipy.stats.pearsonr(features[k1], features[k2])[0]
                S = scipy.stats.spearmanr(features[k1], features[k2])[0]
                #K = scipy.stats.kendalltau(features[k1], features[k2])[0]
                
                if abs(P) > 0.5 or abs(S) > 0.5:
                    print ("\"",k1, "\" vs \"", k2, "\" corr P and S: , ", P, " , ", S)

    for k in features:
        P = scipy.stats.pearsonr(features[k], labels)[0]
        S = scipy.stats.spearmanr(features[k], labels)[0]

        if abs(P) > 0.5 or abs(S) > 0.5:
            print ("\"",k, "\" vs ", labelname, " corr P and S: , ", P, " , ", S)
