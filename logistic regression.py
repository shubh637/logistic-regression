#!/usr/bin/env python
# coding: utf-8

# In[2]:


#logistic regression
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
dataset=pd.read_csv("D:\\Downloads\\user-data.csv")
x=dataset.iloc[:,[2,4]].values
y=dataset.iloc[:,4].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#pre_processing
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)
#classification
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_pred,y_test)
print(cm)


# In[ ]:




