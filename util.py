import pandas as pd
import numpy as nm


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
