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

#1.Gaussion Naive Bayes
#Tahmin etmek istediğimiz veri(column) sürekli bir değer
#reel, ondalıklı bir sayı olabiliyorsa kullanırız.

#2.Multinomial Navie Bayes
#Nominal bir değeri tahmin edeceksek mesela
#erkek mi kadın mı ,arabanızın markası bunlara numara veriyorsak
#label encoder gibi multinomial navie bayesi kullanırız

#3.Bernoulli Naive Bayes
#Bir veya iki tane seçeneğimiz varsa 1 veya 0 gibi
#erkek mi kadın mı ,sigara içiyor mu içmiyor mu gibi
#bernoulli naive bayesi kullanırız

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
gnb=GaussianNB()
bnb=BernoulliNB()
mnb=MultinomialNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
print(y_pred)
print(y_test)
#CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Naive Bayes")
print(cm) 