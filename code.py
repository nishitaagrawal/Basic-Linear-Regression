import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
data=pd.read_csv("D:/Desktop/car_data.csv")
data.head()
X=data["Present_Price"]
Y=data["Selling_Price"]
th0=0
th1=0
alpha=0.01
no_iter=500
for i in range(no_iter):
    S1=0
    for j in range(len(X)):
        a=(th1+th0*X[j]-Y[j])*X[j]
        S1+=a
    S2=0    
    for k in range(len(Y)):
        b=(th1+th0*X[k]-Y[k])
        S2+=b
    th0=th0-(alpha*S1)/len(X)
    th1=th1-(alpha*S2)/len(X)
y=th0*X+th1
plt.scatter(X,Y)
plt.plot(X,y,color='red')
plt.show()
print("r2square is" ,r2_score(Y,y))
