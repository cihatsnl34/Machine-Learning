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
veriler= pd.read_csv("satislar.csv")
#veriler= pd.read_csv("veriler.csv")


aylar=veriler[["Aylar"]]

satislar=veriler[["Satislar"]]


satislar2=veriler.iloc[:,:1].values

#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)

'''
#verilerin ölçeklenmesi aynı dünyaya koyuyoruz verileri
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train= sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)

Y_train= sc.fit_transform(y_train)
Y_test= sc.fit_transform(y_test)

'''
#Model İnşası
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

#Tahmin edilmesi y_testi tahmin ettik
tahmin=lr.predict(x_test)

#İndexine göre küçükten büyüğe sıralama 
x_train=x_train.sort_index()
y_train=y_train.sort_index()
#Basit Doğrusal Regresyon Görselleştirilmesi(Grafik)
plt.scatter(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))


