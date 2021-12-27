# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:30:46 2021

@author: cihat
"""
#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
#2.veriOnisleme
#2.1 veri yükleme

veriler= pd.read_csv("veriler.csv")

cinsiyet=veriler.iloc[:,4:5].values
from sklearn import preprocessing
#1,2,3 :LabelEncoder() 
le=preprocessing.LabelEncoder()
cinsiyet[:,0]=le.fit_transform(veriler.iloc[:,-1])
print(cinsiyet)



x=veriler.iloc[:,1:4].values
y=cinsiyet
#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)


#verilerin ölçeklenmesi aynı dünyaya koyuyoruz verileri
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train= sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)


abc=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=4)
abc.fit(X_train,y_train)
y_pred=abc.predict(X_test)
#CONFUSION MATRIX


cm=confusion_matrix(y_test,y_pred)
print(cm) 
print("ACC")
print(accuracy_score(y_test, y_pred))
print("F2 Score")
print(f1_score(y_test, y_pred))
