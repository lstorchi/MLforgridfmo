
import matplotlib.pyplot as plt
import pandas as pd
import numpy as nm

import scipy.stats
import os.path
import sys

import util

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
        names, features, labels = util.readfile(filename, labelname)
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
                    plt.xlabel(k1)
                    plt.ylabel(k2)
                    plt.scatter(features[k1], features[k2], alpha=0.5)
                    plt.savefig((k1 + "_" + k2 +".png").replace(" ", "_"))
                    plt.close()

    for k in features:
        P = scipy.stats.pearsonr(features[k], labels)[0]
        S = scipy.stats.spearmanr(features[k], labels)[0]

        if abs(P) > 0.5 or abs(S) > 0.5:
            print ("\"",k, "\" vs ", labelname, " corr P and S: , ", P, " , ", S)
