# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:30:46 2021

@author: cihat
"""


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas_profiling
#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)
#Dataset hakkında rapor veriyor.
#print(veriler.profile_report())
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

cm = confusion_matrix(y_test,y_pred)
print(cm)
print("ACC")
print(accuracy_score(y_test, y_pred))
print("F2 Score")
print(f1_score(y_test, y_pred, average=None))

#KNN Classification

#n_neighbors :komşu sayısını veririz yazmazsak eğer default olarak 5 alır 
#metric: komşu ile kendi arasında ki mesafeyi ölçecek metriği belirtiriz ona göre ölçer.
#sklearn.neigbors dökümantasyonunda detayları var
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
"""
print(classifier.predict(sc.transform([[30,87000]])))
"""
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("ACC")
print(accuracy_score(y_test, y_pred))
print("F2 Score")
print(f1_score(y_test, y_pred, average=None))

#SVM Classification
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)
print("ACC")
print(accuracy_score(y_test, y_pred))
print("F2 Score")
print(f1_score(y_test, y_pred, average=None))
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
print("ACC")
print(accuracy_score(y_test, y_pred))
print("F2 Score")
print(f1_score(y_test, y_pred, average=None))

#Decision Tree Classification

#default olarak gini fonksiyonunu alır logaritma hesabını
#yapmadan formule koyar
#entropy log olarak formüle koyup yapıyor 
#Sonucu çok etkilemez
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)
print("ACC")
print(accuracy_score(y_test, y_pred))
print("F2 Score")
print(f1_score(y_test, y_pred, average=None))
#Random Forest Classification

#default olarak gini fonksiyonunu alır logaritma hesabını
#yapmadan formule koyar
#entropy log olarak formüle koyup yapıyor 
#Sonucu çok etkilemez
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)
print("ACC")
print(accuracy_score(y_test, y_pred))
print("F2 Score")
print(f1_score(y_test, y_pred, average=None))

    
# 7. ROC , TPR, FPR değerleri 

#False Positive 
y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)

"""Görselleştirme

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""







