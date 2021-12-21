# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 17:57:52 2021

@author: cihat
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

df=pd.read_csv('sepet.csv', header=None)
df.head()
t=[]
for i in range(0,7501):
    t.append([str(df.values[i,j]) for j in range (0,20)])
    
kurallar=apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2)
print(list(kurallar))
