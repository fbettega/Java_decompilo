# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:07:05 2019

@author: bettega
"""
#remplacer '//' par une variables files sep 
from pathlib import Path
import numpy as np
import pandas as pd
import os
import re 
from sklearn import preprocessing
from keras.preprocessing import sequence

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

def harmonise_compilo(colone_dinteret,colone_supr):#retireles partie différente du nom du decompilo
    df_list=[]
    for i in range(len(colone_dinteret)):
        temp=re.search('(.+?)/(.+?)/',colone_supr[i])
        ligne=colone_dinteret[i].replace(temp.group(1),'').replace(temp.group(2),'')
        df_list.append(ligne)
    df_list=[word.replace('_','') for word in df_list]
    return(df_list)   
 
current_dir = Path(__file__).parent 
data_folder = current_dir/ "data"
sortie_folder = current_dir/ 'sortie'
sav_temp_folder = current_dir/ 'sortie'/ 'save_temp_file'

sequences_path=list_files(data_folder / "sequences","sequences")
report_path=list_files(data_folder / "report_prefix","report_prefix")



report_df=read_files_path(report_path,"Y")
sequence_df=read_files_path(sequences_path,"X")    

compilo=report_path
compilo = [w.replace(str(data_folder / "report_prefix")+'/','') for w in compilo]
compilo = [w.replace('_report.csv','') for w in compilo]
for i in range(len(report_df)):    
    report_df[i]['Compilo']=[compilo[i]]*report_df[i].shape[0]




report_une_table= pd.concat(report_df, join='inner').reset_index(drop=True) #regrouppement de tout les noms de fichiers pour lessquels j'ai un Y
matchers = report_une_table['Class'].values.tolist() #passage en liste
matching = [s for s in sequences_path if any(xs in s for xs in matchers)]#recherche des éléements matchant les paths de sequences  non vide          
matching = [w.replace(str(data_folder / "sequences")+'/','') for w in matching] #retrait du début du chemins pour faciliter la lecture


def jonction_data(x):
    for i in matching:
        if i in x:
            return matching.index(i)
    else:
        return np.nan     
report_une_table['Match'] = report_une_table['Class'].apply(jonction_data)

#encodage en sequences
dico_instruc=[x for xs in sequence_df for x in xs]#création d'un vector contenant toutes les instructions
le = preprocessing.LabelEncoder()
le.fit(dico_instruc)
X_plus_un=[i+1 for i in [le.transform(item) for item in sequence_df]]#ajout de 1 en previsions de sequences pad qui met des zero ajouté par pad.sequence pour obtenir des seqiences de même taille
X = sequence.pad_sequences(X_plus_un)

report_une_table['Compilo']=harmonise_compilo(report_une_table['Compilo'],report_une_table['Class'])


report_une_table_temp=report_une_table[report_une_table.Match.notnull()]#retirer packages info et tga cts


report_une_table_temp = report_une_table_temp.pivot_table(index=['Class','Match'],columns='Compilo',values='isRecompilable', aggfunc='first').reset_index().set_index(['Class'])
Y=report_une_table_temp[report_une_table_temp.columns.difference(['Match'])]
jonction=report_une_table_temp['Match']




with open(sav_temp_folder / 'report_path.txt', 'w') as f:
    for item in report_path:
        f.write("%s\n" % item)
with open(sav_temp_folder / 'sequences_path.txt', 'w') as f:
    for item in sequences_path:
        f.write("%s\n" % item)       

pd.DataFrame(sequence_df).to_csv(sav_temp_folder /'sequence_df.csv')
report_une_table.to_csv(sav_temp_folder /sav_temp_folder /'report_une_table.csv')
np.savetxt(sortie_folder /'X.csv', X, fmt='%i', delimiter=",")
Y.to_csv(sortie_folder/'Y.csv')

















            