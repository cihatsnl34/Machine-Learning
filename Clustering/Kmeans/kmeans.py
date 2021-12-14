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

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4,init='k-means++')

kmeans.fit(X)
#Oluşturulan merkez noktalar
print(kmeans.cluster_centers_)

#WCSS değerini buluyoruz Grafikte kırılma noktasına göre n_clusters değerini alıyoruz
#k değeri bulma = n_clusters değerini bulma
"""sonuclar = []
for i in range(1,10):
    kmeans=KMeans(n_clusters=i , init='k-means++', random_state=100)
    kmeans.fit(X)
    #WCSS değerini getiren fonksiyon:inertia_
    sonuclar.append(kmeans.inertia_)
plt.plot(range(1,10), sonuclar)"""
#Hangi sınıfta olduğunu buluyoruz burada
print(kmeans.predict([[60000,1000]]))
