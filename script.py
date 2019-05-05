# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:07:05 2019

@author: bettega
"""
#remplacer '//' par une variables files sep 

import numpy as np
import pandas as pd
import os
#import re #plus nécessaire
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


def list_files(root,path):
    files=[os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files]
    return(files)
         
def read_files_X(path):
    with open (path, "r") as myfile:
        data = myfile.read().splitlines()
    return(data)    
    
def read_files_path(files_path,methode):
    df_list = []
    for files in files_path:
        if methode=="Y":
            df = pd.read_csv(files)
        elif methode=="X":
            #regex = re.compile('#\d')#création de la methode annotation pour remplacer cette anomalie
            #regex2 = re.compile('^[0-9]')#Suppression car toujours precédé de look-up switch                   
            df = read_files_X(files)
            #df = [i for i in  df if not regex.search(i)]
            #df = [i for i in  df if not regex2.search(i)]
        df_list.append(df)
    return(df_list)   

   



sequences_path=list_files("c:\\Users\\bettega\\Desktop\\nico\\papier2\\test_réseau\\data\\sequences\\","sequences")

report_path=list_files("c:\\Users\\bettega\\Desktop\\nico\\papier2\\test_réseau\\data\\report_prefix\\","report_prefix")




report_df=read_files_path(report_path,"Y")
sequence_df=read_files_path(sequences_path,"X")    

empty_instruc=[i for i,x in enumerate(sequence_df) if not x]


sequence_df_non_vide = [i for j, i in enumerate(sequence_df) if j not in empty_instruc]

sequence_path_non_vide = [i for j, i in enumerate(sequences_path) if j not in empty_instruc]





report_une_table= pd.concat(report_df, join='inner')#regrouppement de tout les noms de fichiers pour lessquels j'ai un Y
matchers = report_une_table['Class'].values.tolist() #passage en liste
matchers=[w.replace('/','\\') for w in matchers]#remplacement dans la chaine de charactères pour matcher sequences
matching = [s for s in sequence_path_non_vide if any(xs in s for xs in matchers)]#recherche des éléements matchant les paths de sequences  non vide          
matching = [w.replace('c:\\Users\\bettega\\Desktop\\nico\\papier2\\test_réseau\\data\\sequences\\','') for w in matching] #retrait du début du chemins pour faciliter la lecture
matching_slash=[w.replace('\\','/') for w in matching]

def jonction_data(x):
    for i in matching_slash:
        if i in x:
            return matching_slash.index(i)
    else:
        return np.nan     
report_une_table['Match'] = report_une_table['Class'].apply(jonction_data) 

#================================================================================







#================================================================================
        
#encodage en sequences
from sklearn import preprocessing
dico_instruc=[x for xs in sequence_df_non_vide for x in xs]#création d'un vector contenant toutes les instructions
le = preprocessing.LabelEncoder()
le.fit(dico_instruc)
X=[le.transform(item) for item in sequence_df_non_vide]
X_plus_un=[i+1 for i in X]#ajout de 1 en previsions de sequences pad qui met des zero ajouté par pad.sequence pour obtenir des seqiences de même taille
X_final = sequence.pad_sequences(X_plus_un)

#le.classes_
# le.inverse_transform([0, 0, 1, 2]) fonction pour revenir au instruction pensez a faire moins un



X_train,X_test= train_test_split(X_final, test_size=0.33)







#encodage en compte ou en frequences

t = Tokenizer()
t.fit_on_texts(sequence_df)
encoded_docs = t.texts_to_matrix(sequence_df, mode='count')

# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)


print(encoded_docs)














#================================================================================

report_une_table= pd.concat(report_df, join='inner')#regrouppement de tout les noms de fichiers pour lessquels j'ai un Y
matchers = report_une_table['Class'].values.tolist() #passage en liste
matchers=[w.replace('/','\\') for w in matchers]#remplacement dans la chaine de charactères pour matcher sequences
matching = [s for s in sequences_path if any(xs in s for xs in matchers)]#recherche des éléements matchant les paths de sequences  non vide          
matching = [w.replace('c:\\Users\\bettega\\Desktop\\nico\\papier2\\test_réseau\\data\\sequences\\','') for w in matching] #retrait du début du chemins pour faciliter la lecture
matching_slash=[w.replace('\\','/') for w in matching]

report_une_table['Match'] = report_une_table['Class'].apply(jonction_data) 

report_une_table['Match'].count() 

na_df=report_une_table[report_une_table.Match.notnull()==False]
na_df.to_csv(r'c:\\Users\\bettega\\Desktop\\nico\\papier2\\test_réseau\\bob.csv')
















            