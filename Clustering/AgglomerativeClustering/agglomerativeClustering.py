# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:27:05 2021

@author: cihat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

veriler = pd.read_csv("musteriler.csv")

X=veriler.iloc[:,3:].values

from sklearn.cluster import AgglomerativeClustering
#Affinity=Kümeler arası ölçüm metriği oklid olarak yaptık default olarak da oklid geliyor
#linkage :Hangi bağlantı kriterini kullanmalıyım. Bağlantı kriteri, gözlem kümeleri arasında hangi mesafenin kullanılacağını belirler. Algoritma, bu ölçütü en aza indiren küme çiftlerini birleştirecektir.
ac= AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
y_tahmin=ac.fit_predict(X)
print(y_tahmin)


#Kümelerin grafikleştirilmesi
label=ac.fit_predict(X)
u_labels = np.unique(label)
 

 
for i in u_labels:
    plt.scatter(X[label == i , 0] , X[label == i , 1] , label = i)

#legend : grafikle ilgili bilgi verir Şu renk=1 dir gibi
plt.legend()
plt.show()

import scipy.cluster.hierarchy as sch
#dendogram kmeans de ki WCSS gibi n_clusters değeri bulmaya yarıyor 
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()