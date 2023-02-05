#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:19:36 2023

@author: zhongmeiru
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

    
def truncate(n,decimals = 2):
    multiplier = 10 ** decimals
    return (int(n * multiplier)/multiplier)
    

##Understand the dataset
df = pd.read_csv("german_credit_data.csv",index_col=0)
print(df.info())
print(df.nunique())
print(df.head())

#%%Explore the dataset
#First, plot the target Risk distribution
df['Risk'].value_counts().plot(kind='bar')
dfgood = df[df['Risk'] == 'good']
dfbad = df[df['Risk'] == 'bad']

#Age
plt.hist(df['Age'],bins=10,alpha=0.5)
plt.hist(dfgood['Age'],bins=10,alpha=0.5)
plt.hist(dfbad['Age'],bins=10,alpha=0.5)

#Create new variable based on age segments
interval = (18,25,35,60,120)
agegroup = ['Student','Young','Adult','Senior']
df['Agegroup'] = pd.cut(df.Age,interval,labels=agegroup)
df['Agegroup'].value_counts().plot(kind='bar')

#updated subdataset
dfgood = df[df['Risk'] == 'good']
dfbad = df[df['Risk'] == 'bad']

#Housing 
df['Housing'].value_counts().plot(kind='bar')
dfgood['Housing'].value_counts().plot(kind='bar')
dfbad['Housing'].value_counts().plot(kind='bar')

#Sex
df['Sex'].value_counts().plot(kind='bar')
dfgood['Sex'].value_counts().plot(kind='bar')
dfbad['Sex'].value_counts().plot(kind='bar')

#Crosstable
print(df.columns)
print(pd.crosstab(df['Checking account'], df['Sex']))
print(pd.crosstab(df['Purpose'], df['Sex']))
print(pd.crosstab(df['Risk'], df['Sex']))
print(pd.crosstab(df['Risk'], df['Agegroup']))

#unique values in each variable
print('Purpose: ',df['Purpose'].unique())
print('Sex: ',df['Sex'].unique())
print('Housing: ',df['Housing'].unique())
print('Saving accounts: ',df['Saving accounts'].unique())
print('Risk: ',df['Risk'].unique())
print('Checking account: ',df['Checking account'].unique())
print('Agegroup: ',df['Agegroup'].unique())

#%%feature engineering - transform to numeric dataframe
print(df.isna().sum())
df['Checking account'] = df['Checking account'].fillna('None')
df['Saving accounts'] = df['Saving accounts'].fillna('None')

purpose = pd.get_dummies(df['Purpose'],prefix='Purpose')
sex = pd.get_dummies(df['Sex'],prefix='Sex')
housing = pd.get_dummies(df['Housing'],prefix='Housing')
risk = pd.get_dummies(df['Risk'],prefix='Risk')
saving = pd.get_dummies(df['Saving accounts'],prefix='Saving accounts')
checking = pd.get_dummies(df['Checking account'],prefix ='Checking account')
agegp = pd.get_dummies(df['Agegroup'],prefix='Agegroup')

df = df.merge(purpose,left_index=True,right_index=True)
df = df.merge(sex,left_index=True,right_index=True)
df = df.merge(housing,left_index=True,right_index=True)
df = df.merge(risk,left_index=True,right_index=True)
df = df.merge(saving,left_index=True,right_index=True)
df = df.merge(checking,left_index=True,right_index=True)
df = df.merge(agegp,left_index=True,right_index=True)

del df['Purpose']
del df['Sex']
del df['Housing']
del df['Saving accounts']
del df['Risk']
del df['Checking account']
del df['Agegroup']
del df['Risk_good']

#%%Correlation heatmap
sns.heatmap(df.astype(float).corr(),linewidth =0.1,vmax = 1.0,square=True,linecolor='white',annot=True)
plt.show()

#%%Preprocessing
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,fbeta_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

df['Credit amount'] =np.log(df['Credit amount'])
print(df.columns)
X = df.drop('Risk_bad',1).values
Y = df['Risk_bad'].values

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size = 0.25,random_state=20)

seed=7
models = []
models.append(('LR',LogisticRegression()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('RF',RandomForestClassifier()))
models.append(('SVM',SVC(gamma='auto')))
models.append(('XGB',XGBClassifier()))

results = []
names = []
scoring ='recall'

for name, model in models:
    kfold = KFold(n_splits=10,random_state=seed)
    cv_results = cross_val_score(model, X_train,Y_train,cv=kfold,scoring = scoring)
    results.append(cv_results)
    names.append(name)
    print(name," mean: ", truncate(cv_results.mean()),", std: ",truncate(cv_results.std()))

#the best model is GaussianNB, followed by XGB,CART, LR, RF

#%% Random Forest to predict target
param_grid = {'max_depth':[3,5,7,10,None],'n_estimators':[3,5,10,25,50,150],'max_features': [4,7,15,20]}

model = RandomForestClassifier(random_state=2)
grid_search = GridSearchCV(model, param_grid=param_grid,cv=5,scoring='recall')
grid_search.fit(X_train,Y_train)
print(truncate(grid_search.best_score_))
print(grid_search.best_params_)
#0.44
#{'max_depth': None, 'max_features': 15, 'n_estimators': 5}

rf = RandomForestClassifier(max_depth=None, max_features=15, n_estimators=5,random_state=2)
rf.fit(X_train,Y_train)
Y_predict =rf.predict(X_test)
print(truncate(accuracy_score(Y_test, Y_predict)))
print('\n')
print(confusion_matrix(Y_test, Y_predict))
print('\n')
print(truncate(fbeta_score(Y_test, Y_predict,beta=2)))
#0.67; 0.41, not good 

#%% Gaussian model to predict target
from sklearn.utils import resample
from sklearn.metrics import roc_curve

GNB =GaussianNB()
model = GNB.fit(X_train,Y_train)
print('Training score: ')
print(model.score(X_train,Y_train))
#0.66
Y_predict = model.predict(X_test)
print(truncate(accuracy_score(Y_test, Y_predict)))
print('\n')
print(confusion_matrix(Y_test, Y_predict))
print('\n')
print(truncate(fbeta_score(Y_test, Y_predict,beta=2)))
#0.65; 0.65, better results

y_pred_prob = model.predict_proba(X_test)[:,1]
#probability of True Class

fpr,tpr,thresholds = roc_curve(Y_test, y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#%% Model Pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
features = []
features.append(('pca',PCA(n_components=2)))
features.append(('select_best',SelectKBest(k=6)))
feature_union = FeatureUnion(features)

estimators =[]
estimators.append(('feature_union',feature_union))
estimators.append(('logistic',GaussianNB()))
model = Pipeline(estimators)

seed=7
kfold = KFold(n_splits=10,random_state=seed)
results = cross_val_score(model, X_train,Y_train,cv=kfold)
print(truncate(results.mean()))
#0.71

model.fit(X_train,Y_train)
Y_predict = model.predict(X_test)
print(truncate(accuracy_score(Y_test, Y_predict)))
print('\n')
print(confusion_matrix(Y_test, Y_predict))
print('\n')
print(truncate(fbeta_score(Y_test, Y_predict,beta=2)))
#0.72; 0.60, good

#%%XGBoost
#Seting the Hyper Parameters
param_test1 = {
 'max_depth':[3,5,6,10],
 'min_child_weight':[3,5,10],
 'gamma':[0.0, 0.1, 0.2, 0.3, 0.4],
# 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 10],
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}

#Creating the classifier
model_xg = XGBClassifier(random_state=2)

grid_search = GridSearchCV(model_xg, param_grid=param_test1, cv=5, scoring='recall')
grid_search.fit(X_train, Y_train)
grid_search.best_score_
grid_search.best_params_

Y_predict = grid_search.predict(X_test)
print(truncate(accuracy_score(Y_test, Y_predict)))
print('\n')
print(confusion_matrix(Y_test, Y_predict))
print('\n')
print(truncate(fbeta_score(Y_test, Y_predict,beta=2)))
#0.69;0.44, not good

#%%Conclusion
#GaussianNB is the best predictive model with the highest F score 0.65.

