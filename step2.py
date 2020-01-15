
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

    somemissing = False
    for k in featuresselected:
        if k not in features:
            print ("Error cannot find ", k)
            somemissing = True

    if not somemissing:
        print("All the features have been found")

        toberemoved = []

        for allf in features:
            if allf not in featuresselected:
                toberemoved.append(allf)

        for ftorm in toberemoved:
            del features[ftorm]

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

        for idxk1 in range(len(features.keys())):
            for idxk2 in range(idxk1+1, len(features.keys())):
      
                k1 = list(features.keys())[idxk1]
                k2 = list(features.keys())[idxk2]
                if k1 != k2:
                    P = scipy.stats.pearsonr(features[k1], features[k2])[0]
                    S = scipy.stats.spearmanr(features[k1], features[k2])[0]
                    
                    if abs(P) > 0.5 or abs(S) > 0.5:
                        print ("\"",k1, "\" vs \"", k2, "\" corr P and S: , ", P, " , ", S)
      
        for k in features:
            P = scipy.stats.pearsonr(features[k], labels)[0]
            S = scipy.stats.spearmanr(features[k], labels)[0]
      
            if abs(P) > 0.5 or abs(S) > 0.5:
                print ("\"",k, "\" vs ", labelname, " corr P and S: , ", P, " , ", S)

        
        features_array = np.zeros((len(labels), len(features.keys())))

        i = 0
        for k in features:
            features_array[:, i] = features[k]
            i = i + 1

        # elbow method to select optmial number of cluster 

        print ("Start KMeans...")

        distortions = []
        for i in range(1, int(len(labels)/10)):
            km = KMeans(
                    n_clusters=i, init='random',
                    n_init=10, max_iter=300,
                    tol=1e-04, random_state=0)
            km.fit(features_array)
            distortions.append(km.inertia_)


        plt.plot(range(1, int(len(labels)/10)), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.savefig("elbow_distorsion_vs_numofcluster.png")

        # set number of cluster to 4
        NUMOFCLUSTER = 4
        NUM_ITER = 20

        km = KMeans(
                n_clusters=NUMOFCLUSTER, init='random',
                n_init=10, max_iter=300,
                tol=1e-04, random_state=0)
        km.fit(features_array)
        cents = km.cluster_centers_
        print('Centroids:', km.cluster_centers_)

        for iter in range(NUM_ITER):
            km = KMeans(
                n_clusters=NUMOFCLUSTER, init=cents,
                n_init=1)
            km.fit(features_array)
            cents = km.cluster_centers_
            print('Iteration: ', iter)
            print('Centroids:', km.cluster_centers_)


        for i in range(len(labels)):
            print (names[i], " , " , km.labels_[i], " , ", labels[i])

