# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:30:46 2021

@author: cihat
"""


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_excel('Iris.xls')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x = veriler.iloc[:,0:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Logistics Classification
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

#Confusuion_Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Logistic")
print(cm)


#KNN Classification

#n_neighbors :komşu sayısını veririz yazmazsak eğer default olarak 5 alır 
#metric: komşu ile kendi arasında ki mesafeyi ölçecek metriği belirtiriz ona göre ölçer.
#sklearn.neigbors dökümantasyonunda detayları var
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("KNN")
print(cm)
print(knn.predict([[7.2,2.5,6.4,2.2]]))

#SVM Classification
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm1 = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm1)


#Naive Bayes Classification

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

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)


#Decision Tree Classification

#default olarak gini fonksiyonunu alır logaritma hesabını
#yapmadan formule koyar
#entropy log olarak formüle koyup yapıyor 
#Sonucu çok etkilemez
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'gini')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

#Random Forest Classification

#default olarak gini fonksiyonunu alır logaritma hesabını
#yapmadan formule koyar
#entropy log olarak formüle koyup yapıyor 
#Sonucu çok etkilemez
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=9, criterion = 'gini')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


"""
# 7. ROC , TPR, FPR değerleri 

#False Positive 
y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='Iris-setosa')
print(fpr)
print(tpr)



"""

