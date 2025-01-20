import pandas as pd
import joblib
from joblib import load
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import numpy as np

X_train_fin=pd.read_csv('Dataset/X_train.csv')
X_test_fin=pd.read_csv('Dataset/X_test.csv')
y_train=pd.read_csv('Dataset/y_train.csv')
y_test=pd.read_csv('Dataset/y_test.csv')

X=pd.read_csv('Dataset/EasyOCR+clean.csv')
X=X.drop(columns="sample_type")
X=X.drop(columns='ad_screenshot')
X=X.drop(columns='page_screenshot')
names=X.columns.tolist()
best=load('Resultat/Model_GR_Best.joblib')


result = permutation_importance(best, X_test_fin, y_test, n_repeats=10, random_state=42)

importance_means = result['importances_mean']
importance = importance_means[:-(23+768)]

Bert = importance_means[-768:]

first_values = importance_means[:-768]
onehotencoder_var = first_values[-23:]
name_var = onehotencoder_var[:14] #name
TLD_var = onehotencoder_var[14:23] # TLD

mean_bert = np.mean(Bert)
importance = np.append(importance, mean_bert)

mean_name_var = np.mean(name_var)
importance=np.insert(importance,1, mean_name_var)

mean_TLD_var = np.mean(TLD_var)
importance=np.insert(importance,6, mean_TLD_var)
print(len(importance))
print(len(names))
names[3]='Word2Vec'
names[21]='BERT'
import os
if not os.path.exists('./plot'):
    os.makedirs('./plot')
plt.figure(figsize=(10, 6))
plt.barh(range(len(importance)), importance, align='center')
plt.yticks(range(len(importance)), names)
plt.xlabel('permutation_importance')
plt.title(f'permutation_importance des features pour le Meilleur modèle ')

plt.savefig(f'./plot/Feature_Importance_Best_Model.png')  
plt.close()  