# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:30:46 2021

@author: cihat
"""
#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#2.veriOnisleme
#2.1 veri yükleme

veriler= pd.read_csv("veriler.csv")


x=veriler.iloc[:,1:4].values
y=veriler.iloc[:,4:5].values
#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)


#verilerin ölçeklenmesi aynı dünyaya koyuyoruz verileri
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train= sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)


from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(x_train,y_train)
y_pred=logr.predict(x_test)
print(y_pred)
print(y_test)
print(logr.predict([[185,60,19]]))
#CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm) 