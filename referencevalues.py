import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from scipy.stats import pearsonr




boston=load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)
boston_df['Price']=boston.target
newX=boston_df.drop('Price',axis=1)
print(newX[0:3]) # check 
newY=boston_df['Price']

#print type(newY)# pandas core frame

X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)
print(len(X_test), len(y_test))
lr = LinearRegression()
lr.fit(X_train, y_train)

rr = Ridge(alpha=0.01) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely
# restricted and in this case linear and ridge regression resembles
rr.fit(X_train, y_train)

rr100 = Ridge(alpha=100) #  comparison with alpha value
rr100.fit(X_train, y_train)

train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)

y_pred = lr.predict(X_test) 

lrrms = math.sqrt(mean_squared_error(y_test, y_pred))
lrmae = mean_absolute_error(y_test, y_pred)
absdiff = [abs(x - y) for x, y in zip(y_test, y_pred)]
lrmaxae = np.amax(absdiff)
lrrp = pearsonr(y_test, y_pred)[0]

y_pred = lr.predict(X_train) 

lrrmstrain = math.sqrt(mean_squared_error(y_train, y_pred))
lrmaetrain = mean_absolute_error(y_train, y_pred)
absdiff = [abs(x - y) for x, y in zip(y_train, y_pred)]
lrmaxaetrain = np.amax(absdiff)
lrrptrain = pearsonr(y_train, y_pred)[0]

print("linear regression")
print("  train score:", train_score)
print("  test  score:", test_score)
print("  Mean   RMSE: %10.5f"%(lrrms))
print("  Mean    MAE: %10.5f"%(lrmae))
print("  Mean  MaxAE: %10.5f"%(lrmaxae))
print("  Mean     rP: %10.5f"%(lrrp))
print("  Mean   RMSE train: %10.5f"%(lrrmstrain))
print("  Mean    MAE train: %10.5f"%(lrmaetrain))
print("  Mean  MaxAE train: %10.5f"%(lrmaxaetrain))
print("  Mean     rP train: %10.5f"%(lrrptrain))


Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)

y_pred = rr.predict(X_test) 

lrrms = math.sqrt(mean_squared_error(y_test, y_pred))
lrmae = mean_absolute_error(y_test, y_pred)
absdiff = [abs(x - y) for x, y in zip(y_test, y_pred)]
lrmaxae = np.amax(absdiff)
lrrp = pearsonr(y_test, y_pred)[0]

y_pred = rr.predict(X_train) 

lrrmstrain = math.sqrt(mean_squared_error(y_train, y_pred))
lrmaetrain = mean_absolute_error(y_train, y_pred)
absdiff = [abs(x - y) for x, y in zip(y_train, y_pred)]
lrmaxaetrain = np.amax(absdiff)
lrrptrain = pearsonr(y_train, y_pred)[0]

print("Ridge regression")
print("  train score:", Ridge_train_score)
print("  test  score:", Ridge_test_score)
print("  Mean   RMSE: %10.5f"%(lrrms))
print("  Mean    MAE: %10.5f"%(lrmae))
print("  Mean  MaxAE: %10.5f"%(lrmaxae))
print("  Mean     rP: %10.5f"%(lrrp))
print("  Mean   RMSE train: %10.5f"%(lrrmstrain))
print("  Mean    MAE train: %10.5f"%(lrmaetrain))
print("  Mean  MaxAE train: %10.5f"%(lrmaxaetrain))
print("  Mean     rP train: %10.5f"%(lrrptrain))

Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)

y_pred = rr100.predict(X_test) 

lrrms = math.sqrt(mean_squared_error(y_test, y_pred))
lrmae = mean_absolute_error(y_test, y_pred)
absdiff = [abs(x - y) for x, y in zip(y_test, y_pred)]
lrmaxae = np.amax(absdiff)
lrrp = pearsonr(y_test, y_pred)[0]

y_pred = rr100.predict(X_train) 

lrrmstrain = math.sqrt(mean_squared_error(y_train, y_pred))
lrmaetrain = mean_absolute_error(y_train, y_pred)
absdiff = [abs(x - y) for x, y in zip(y_train, y_pred)]
lrmaxaetrain = np.amax(absdiff)
lrrptrain = pearsonr(y_train, y_pred)[0]

print("Ridge 100 regression")
print("  train score:", Ridge_train_score100)
print("  test  score:", Ridge_test_score100)
print("  Mean   RMSE: %10.5f"%(lrrms))
print("  Mean    MAE: %10.5f"%(lrmae))
print("  Mean  MaxAE: %10.5f"%(lrmaxae))
print("  Mean     rP: %10.5f"%(lrrp))
print("  Mean   RMSE train: %10.5f"%(lrrmstrain))
print("  Mean    MAE train: %10.5f"%(lrmaetrain))
print("  Mean  MaxAE train: %10.5f"%(lrmaxaetrain))
print("  Mean     rP train: %10.5f"%(lrrptrain))

#plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers
#plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency
#plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
#plt.xlabel('Coefficient Index',fontsize=16)
#plt.ylabel('Coefficient Magnitude',fontsize=16)
#plt.legend(fontsize=13,loc=4)
#plt.show()


