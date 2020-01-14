
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


