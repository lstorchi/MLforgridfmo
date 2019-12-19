import pandas as pd
import sys

#######################################################################

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

#######################################################################

def difference(lst1, lst2):
    return list(set(lst1) - set(lst2))

#######################################################################

filename = ""

if len(sys.argv) != 2:
    print("usage: " , sys.argv[0] , " file.xslx")
    exit(1)
else:
    filename = sys.argv[1]

xls = pd.ExcelFile(filename)

names = []
features = []

thefirst = True
for name in xls.sheet_names:
    print ("Parsing data from ", name)
    df = xls.parse(name)
    ft = list(df.columns.values)
    print("  reading names from ", ft[0])
    if thefirst:
        names = list(df[ft[0]].values)
        thefirst = False
    else:
        localnames = list(df[ft[0]].values)
        if len(names) != len(localnames):
            print("  set of names is different ")
            print("  ", len(names) , " vs ", len(localnames))
            print("  ", difference(localnames, names))
            print("  ", difference(names, localnames))
            exit(1)

