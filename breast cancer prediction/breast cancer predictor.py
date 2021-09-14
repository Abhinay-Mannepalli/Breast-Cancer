import pandas as pd  
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import cv2
import numpy as np
import pandas as pd
import csv

df = pd.read_csv('dataset_new.csv')
features = df[['radmean','textmean','perimean','areamean','smoomean','compmean','concave','copoints','symmetry','fractal',
'radse','textse','perimtrse','arease','smoothse','compse','concavse','conptse']].values
labels = df[['result']].values

a=float(input("enter radius_mean"))
b=float(input("enter texture_mean"))
c=float(input("enter perimeter_mean"))
d=float(input("enter area_mean"))
e=float(input("enter smoothness_mean"))
f=float(input("enter comp_mean"))
g=float(input("concave_mean"))
h=float(input("enter concavepoints_mean"))
i=float(input("enter  symmetry_mean"))
j=float(input("enter fractal_mean"))
k=float(input("enter radius standard error"))
l=float(input("enter texture standard error"))
m=float(input("enter perimeter standard error"))
n=float(input("enter area standard error"))
o=float(input("enter smoothness se"))
p=float(input("enter comp_standard error"))
q=float(input("enter concave standard error"))
r=float(input("enter concave points error"))

#'''used for prediction'''
rf=RandomForestClassifier()

rf.fit(features,labels)
#'''used to train the model'''
pred_rf = rf.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]])
#'''to predict'''
print("\n species no: ",pred_rf)








