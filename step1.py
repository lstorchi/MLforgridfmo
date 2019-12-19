import pandas as pd
import sys

filename = ""

if len(sys.argv) != 2:
    print("usage: " , sys.argv[0] , " file.xslx")
    exit(1)
else:
    filename = sys.argv[1]

xls = pd.ExcelFile(filename)

features = []

for name in xls.sheet_names:
    print ("Parsing data from ", name)
    df = xls.parse(name)
    ft = list(df.columns.values)
    print("  reading names from ", ft[0])
    print(df[ft[0]].values)

