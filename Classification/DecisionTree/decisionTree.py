# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:30:46 2021

@author: cihat
"""
#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix

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

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
#default olarak gini fonksiyonunu alır logaritma hesabını
#yapmadan formule koyar
#entropy log olarak formüle koyup yapıyor 
#Sonucu çok etkilemez
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print("DTC")
print(cm)