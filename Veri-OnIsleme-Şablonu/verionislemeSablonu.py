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
veriler= pd.read_csv("eksikveriler.csv")
#veriler= pd.read_csv("veriler.csv")


#2.2 Eksik veriler
#sci - kit learn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
Yas = veriler.iloc[:,1:4].values
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)

#Encoder:Kategorik -> Numeric

ulke=veriler.iloc[:,0:1].values
print(ulke)
from sklearn import preprocessing
#1,2,3 :LabelEncoder() 
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)


#Kategorik verileri topluca Numeric verilere çevirme
#from sklearn.preprocessing import LabelEncoder
#veriler = veriler.apply(LabelEncoder().fit_transform)

#1-0-0 :OneHotEncoder() 
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)


#numpy dizileri dataframe donusumu
sonuc=pd.DataFrame(data=ulke, index=range(22),columns=['fr','tr','us'])
print(sonuc)
print(veriler)
sonuc2=pd.DataFrame(data=Yas, index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)
cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
print(sonuc3)


#dataframe birleştirme işlemi axis:yanına yerleştiriyor default olarak altına verileri yerleştiriyor
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
s2=pd.concat([s,sonuc3],axis=1)
print(s2)


#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)


#verilerin ölçeklenmesi aynı dünyaya koyuyoruz verileri
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train= sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)


