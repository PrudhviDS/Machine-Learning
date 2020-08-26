#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('oilpipelineincidents.csv')


# In[3]:


df.info()


# In[4]:


data = df.drop(labels = ['Report Number', 'All Costs', 'Other Costs', 'Environmental Remediation Costs', 'Emergency Response Costs', 'Public/Private Property Damage Costs', 'Lost Commodity Costs', 'Property Damage Costs', 'Public Evacuations', 'Restart Date/Time', 'Shutdown Date/Time', 'Net Loss (Barrels)', 'Accident Longitude', 'Accident Latitude', 'Liquid Name', 'Operator ID', 'Accident Date/Time', 'Accident Year', 'Supplemental Number', 'Report Number'], axis = 1)


# In[5]:


data.info()


# In[6]:


temp = data.drop(labels = ['Liquid Recovery (Barrels)', 'Intentional Release (Barrels)', 'Unintentional Release (Barrels)'], axis = 1)


# In[7]:


temp.fillna(method = 'ffill', inplace = True)


# In[8]:


tempf = temp.drop(labels = 'Pipeline Shutdown', axis = 1)


# In[9]:


from sklearn.preprocessing import LabelEncoder
tempf = tempf.apply(LabelEncoder().fit_transform)
tempf


# In[10]:


tempint = data.loc[:,['Liquid Recovery (Barrels)', 'Intentional Release (Barrels)', 'Unintentional Release (Barrels)']]


# In[11]:


tempint.fillna(value = 0, inplace = True)
tempint.isnull().sum()


# In[12]:


x = pd.concat([tempf, tempint], axis = 1)
x.info()


# In[13]:


from scipy.stats import zscore
x = x.apply(zscore)
x.describe()


# In[14]:


y = df.loc[:,'Pipeline Shutdown']
y.fillna(method = 'ffill', inplace = True)


# In[15]:


y.value_counts()


# In[16]:


y = pd.get_dummies(y, drop_first = True)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.7, random_state = 35)


# In[19]:


lreg = LogisticRegression(random_state = 17)
lreg.fit(x_train, y_train)


# In[20]:


pred = lreg.predict(x_test)


# In[21]:


pred


# In[22]:


from sklearn.metrics import accuracy_score


# In[23]:


accuracy_score(y_test, pred)


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


rf = RandomForestClassifier(n_estimators = 100, random_state = 7, criterion = 'gini')


# In[65]:


from sklearn.model_selection import GridSearchCV
rf_param_grid = {"max_depth": [4, 6, 8, 10],
                "max_features": ["sqrt", "log2", 2, 5, 8],
                "min_samples_split": [3, 4, 5, 8],
                "min_samples_leaf": [3, 4, 5, 8],
                "bootstrap": [False, True],
                "n_estimators" :[100],
                "criterion": ["gini", "entropy"],
                "class_weight": [None, 'balanced']
               }
               
# Search grid and store best estimator.
gsRFC = GridSearchCV(rf,param_grid = rf_param_grid, cv=4, scoring="f1", n_jobs= 4)
gsRFC.fit(x_train,y_train)
RFC_best = gsRFC.best_estimator_

# Print best score.
print('Best score: {}'.format(gsRFC.best_score_))
print('Best parameters: {}'.format(gsRFC.best_params_))


# In[62]:


rf.fit(x_train, y_train)


# In[63]:


pred = rf.predict(x_test)
accuracy_score(y_test, pred)


# In[28]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state = 7)
model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy_score(y_test, pred)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state = 14)
model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy_score(y_test, pred)


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
NNH = KNeighborsClassifier(n_neighbors = 4)


# In[31]:


NNH.fit(x_train, y_train)


# In[32]:


pred = NNH.predict(x_test)
accuracy_score(y_test, pred)


# In[33]:


for i in range(1,50):
    NNH = KNeighborsClassifier(n_neighbors= i , weights = 'uniform' )
    NNH.fit(x_train, y_train)
    pred = NNH.predict(x_test)
    NNH.score(x_test, y_test)
    s = accuracy_score(y_test, pred)
    MSE = 1 - s
    print(MSE)
    i = i + 1


# In[34]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB


# In[45]:


nb = GaussianNB()
nb.fit(x_train, y_train)


# In[46]:


pred = nb.predict(x_test)
accuracy_score(y_test, pred)


# In[47]:


from sklearn.svm import SVC
clf = SVC(kernel='linear') 
clf.fit(x_train, y_train)


# In[48]:


pred = clf.predict(x_test)
accuracy_score(y_test, pred)


# In[49]:


from sklearn.linear_model import Lasso, Ridge


# In[50]:


model = Lasso(fit_intercept = True)


# In[51]:


model.fit(x_train, y_train)


# In[54]:


pred = model.predict(x_test)


# In[66]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[67]:


bestfeatures = SelectKBest(score_func=chi2, k=10)


# In[68]:


fit = bestfeatures.fit(x_train,y_train)


# In[69]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x_train,y_train)
print(model.feature_importances_)


# In[ ]:




