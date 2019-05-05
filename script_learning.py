# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:59:09 2019

@author: bettega
"""
import numpy as np
import pandas as pd
import os
import re 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

def read_files_X(path):
    with open (path, "r") as myfile:
        data = myfile.read().splitlines()
    return(data)    
    
report_une_table=pd.read_csv('c:\\Users\\bettega\\Desktop\\nico\\papier2\\test_réseau\\sortie\\report_une_table.csv')
X_final=pd.read_csv('c:\\Users\\bettega\\Desktop\\nico\\papier2\\test_réseau\\sortie\\X_final.csv').values[:,1:]
report_path= read_files_X('c:\\Users\\bettega\\Desktop\\nico\\papier2\\test_réseau\\sortie\\report_path.txt')
sequences_path= read_files_X('c:\\Users\\bettega\\Desktop\\nico\\papier2\\test_réseau\\sortie\\sequences_path.txt')

df_list=[]
for i in range(len(report_une_table['Compilo'])):
    temp=re.search('(.+?)/(.+?)/',report_une_table['Class'][i])
    ligne=report_une_table['Compilo'][i].replace(temp.group(1),'').replace(temp.group(2),'')
    df_list.append(ligne)

report_une_table['Compilo']=df_list
report_une_table['Compilo']=report_une_table['Compilo'].str.replace('_','')









report_une_table_temp=report_une_table[report_une_table.Match.notnull()]#retirer packages info et tga cts


report_une_table_temp = report_une_table_temp.pivot_table(index=['Class','Match'],columns='Compilo',values='isRecompilable', aggfunc='first').reset_index().set_index(['Class'])
Y=report_une_table_temp[report_une_table_temp.columns.difference(['Match'])]
jonction=report_une_table_temp['Match']

#===================================================================================================================================
#appprentissage
Y_1=report_une_table_temp[report_une_table_temp.columns.difference(['Match'])]
Y=report_une_table_temp[report_une_table_temp.columns.difference(['Match'])]['CFR-0.141']


X_train,X_test,Y_train,Y_test= train_test_split(X_final, Y,test_size=0.33)


#===================================================================================================================================
#appprentissage
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import load_model


top_words=X_final.shape[0]#je pense que enfait ce truc est la taille de mon dictionnaire pas la longueur
max_review_length=X_final.shape[1]
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(9, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=64)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))




#model
top_words=199
max_review_length=X_final.shape[1]
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=64)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#model1


model.save('my_model.h5')
#=======================================================================================================================================
#compte occurence
report_une_table_temp.isRecompilable.value_counts()
report_une_table_temp['Compilo'].value_counts()
report_une_table.passTests.value_counts()
#===================================================================================================================================

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#===================================================================================================================================

a=report_une_table['Compilo'].unique().tolist()
b=report_une_table['Compilo'].value_counts()

report_une_table['Class'][0]

