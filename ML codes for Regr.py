# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:00:37 2019

@author: win10
"""

import pandas as pd
data = pd.read_excel('sample.xlsx')
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

print(regressor.coef_)
print(regressor.intercept_)

regressor.predict([[15]])


import pandas as pd
data = pd.read_csv('simp_regression.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state =0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.coef_)
print(regressor.intercept_)
y_pred = regressor.predict(x_test)

## Error (route mean squared error (RMSE))
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
print(mae)
print(rmse)

#visualization in python using matplotlib
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
# or .. plt.plot(x_test,y_pred, color = 'blue')
plt.title('training set')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
## for test data
import matplotlib.pyplot as plt
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
# or .. plt.plot(x_test,y_pred, color = 'blue')
plt.title('test set')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


 ### MULTIPLE REGRESSION ###
import pandas as pd
dataset = pd.read_csv('mul_regression.csv')
x = dataset.iloc[:,:].values
## method one-------------1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

x[:,3]= labelencoder.fit_transform(x[:,3])
ohe = OneHotEncoder(categorical_features = [3]) #3 is the column index
#$ X variable disappears because the values are changed into someother datatype that python coudn't
#$ understand so the "to array " is used
x = ohe.fit_transform(x).toarray()
## method two --------------2 
# another way using "get dummies"
dataset1 = pd.get_dummies(dataset,columns= ['State'])

### find ou VIF to find whether mulitcollinearity is there?
import statsmodels.formula.api as sm
def vif_cal(input_data, dependent_col):
    x_var = input_data.drop([dependent_col],axis = 1)
    xvar_names = x_var.columns
    for i in range (0,len(xvar_names)):
        y = x_var[xvar_names[i]]
        x = x_var[xvar_names.drop(xvar_names[i])]
        rsq = sm.ols("y~x",x_var).fit().rsquared
        vif = round(1/(1-rsq),2)
        print(xvar_names[i],"VIF:",vif)
vif_cal(dataset1,'Profit')

dataset2 = dataset1.drop(['State_Florida'],axis = 1)
vif_cal(dataset2,'Profit')

x = dataset2.iloc[:[0,1,2,4,5]].values
y = dataset2.iloc[:,3].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state =0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

print(sqrt(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))

$$ R - square and adjusted R- square $$

import pandas as pd
dataset = pd.read_csv('mul_regression.csv')
data1 = pd.get_dummies(dataset,['State'])
x = data1.iloc[:,[0,1,2,4,5,6]].values
y = data1.iloc[:,3].values

import statsmodels.formula.api as sm
regressor = sm.OLS(y,x).fit()
regressor.summary()

x1 = x[:,[0,1,2,3,4]]
regressor = sm.OLS(y,x1).fit()
regressor.summary()

## $$ POLYNOMIAL REGTRESSION $$ ###
data11 = pd.read_csv('poly_reg.csv')
x = data11.iloc[:,1:2].values
y = data11.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

import matplotlib.pyplot as plt
plt.scatter(x,y,color = 'blue')
plt.plot(x,lin_reg.predict(x),color = 'red')
plt.show()

lin_reg.predict(8)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg2.predict(x_poly),color = 'blue')
plt.show()

#@@@ LOGISTIC REGRESSION !@@@#
#BINARY CLASSIFICATION
#SIGNOID CUREVE - WHERE THE VALUES LIES EITHER 0-1
#LOG(ODDS)- RATIO OF THE PROBABILITY OF ONE EVENT TO ANOTHER
#LOG(ODDS) IS CALLED AS LOGIT FUNCTION
import pandas as pd
dataset = pd.read_csv('logistic_regression.csv')
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# FIND THE CALSS PROBABILITIES
classifier.predict_proba(x_test)

#TO PREDICT THE CLASS LABELS
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

accuracy = cm[0,0]+cm[1,1]/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

print(acuracy*100)

DECISION TREE:
import pandas as pd
data = pd.read_csv('logistic_regression.csv')

x = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

import pydotplus
from sklearn.tree import export_graphviz

dot = export_graphviz(classifier, out_file= None,filled = True,rounded = True)
graph = pydotplus.graph_from_dot_data(dot)
graph.write_png('sample.org')


ASSOCIATION RULES:
import pandas as pd
data = pd.read('MLRSMBAEX2-DataSet.csV')


--------------------DIMENSIONALITY REDUCTION :----------------------------
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = datasets.load_iris()
x = dataset.data
x = pd.DataFrame(x)

y = dataset.target
y = pd.DataFrame(y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
x = pd.DataFrame(x)

from sklearn.decomposition import PCA
pca = PCA()
x = pca.fit_transform(x)

pca.components_[0]

res = pca.explained_variance_ratio_*100

np.cumsum(pca.explained_variance_ratio_*100)

scores = pd.Series(pca.components_[0])
scores.abs().sort_values(ascending = False)


var = pca.components_[0]
## SCREE PLOT

plt.bar(x=range(1,len(var)+1),height= res)
plt.show()

-------------------------- K- NEAREST NEIGHBOUR-----------------------------
import pandas as pd
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    k = KMeans(i, init = 'k-means++', random_state = 123)
    k.fit(x)
    wcss.append(k.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.show()
    
km = KMeans(n_clusters=5, init= 'k-means++',random_state=123)
y_pred = km.fit_predict(x)

plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s= 100,c='red',label = 'Cluster1')    
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s= 100,c='blue',label = 'Cluster2')    
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],s= 100,c='cyan',label = 'Cluster3')    
plt.scatter(x[y_pred==3,0],x[y_pred==3,1],s= 100,c='black',label = 'Cluster4')    
plt.scatter(x[y_pred==4,0],x[y_pred==4,1],s= 100,c='green',label = 'Cluster5')   

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=300,c = 'yellow',label ='centroids')
plt.legend()
plt.show() 

-------------------------------K-MEANS EX-2----------------------------------------------
import pandas as pd
dataset = pd.read_csv('CC GENERAL.csv')     
import numpy as np               
x = dataset.iloc[:,[1,13]].values
np.nan_to_num(x,copy= False)
print(x)

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    k = KMeans(i, init = 'k-means++', random_state =0)
    k.fit(x)
    wcss.append(k.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.show()
    
km = KMeans(n_clusters=6, init= 'k-means++',random_state=123)
y_pred = km.fit_predict(x)

plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s= 100,c='red',label = 'Cluster1')    
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s= 100,c='blue',label = 'Cluster2')    
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],s= 100,c='cyan',label = 'Cluster3')    
plt.scatter(x[y_pred==3,0],x[y_pred==3,1],s= 100,c='black',label = 'Cluster4')    
plt.scatter(x[y_pred==4,0],x[y_pred==4,1],s= 100,c='green',label = 'Cluster5')   
plt.scatter(x[y_pred==5,0],x[y_pred==5,1],s= 100,c='brown',label = 'Cluster6')   

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=300,c = 'yellow',label ='centroids')
plt.legend()
plt.show() 

--------------------------------ENSEMBLE-----------------------------------------
import pandas as pd
data = pd.read_csv('Churn_Modelling.csv')    
x = data.iloc[:,3:13].values
y = data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
x[:,1]= label.fit_transform(x[:,1])
x[:,2]= label.fit_transform(x[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])                       

x = ohe.fit_transform(x).toarray()
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.25,random_state =0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50,criterion = 'entropy',random_state =0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

from sklearn.model_selection import cross_val_score
acc = cross_val_score(classifier,x_train,y_train,cv=10)
acc.mean()
acc.std()
-----------------------------GRID SEARCH---------------------------------------
from sklearn.model_selection import GridSearchCV

param_grid = {'bootstrap':[True],'n_estimators':[10,20,50,100]}
classifier_grid = RandomForestClassifier()
gr = GridSearchCV(classifier_grid,param_grid,cv=10,n_jobs=-1)
gr.fit(x_train,y_train)
gr.best_params_
gr.best_estimator_
-----------------------------XGBOOST-------------------------------------------
from xgboost.sklearn import XGBClassifier

classifier1 = XGBClassifier()
classifier1.fit(x_train,y_train)

y_pred = classifier1.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

accuracy_xgb = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
print(accuracy)
print(accuracy_xgb)
-----------------------------------Hierarchial Clustering--------------------------------------------
import pandas as pd
data = pd.read_csv('Mall_Customers.csv')
x = data.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
dendrogram = sch.dendrogram(sch.linkage(x,method = 'ward'))

plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(x)

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s= 100,c='red',label = 'Cluster1')    
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s= 100,c='blue',label = 'Cluster2')    
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s= 100,c='cyan',label = 'Cluster3')    
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s= 100,c='black',label = 'Cluster4')    
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s= 100,c='green',label = 'Cluster5')   
   
-------------------------------------------------------------------------------------------------------


















