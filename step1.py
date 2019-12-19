import pandas as pd
import sys

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
                print("  set of names is different")
                exit(1)

