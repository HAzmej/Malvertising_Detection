import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib 
import torch
import json

dataset=pd.read_csv('Dataset/EasyOCR+clean.csv')


###############################################################################  Visualisation ##################################################################################
from plugins.vizualiser import show_dataset, heatmap
#show_dataset(dataset)
print("Visualisation termine")

##heatmap(dataset)
print("Heatmap termine")

###############################################################################  Process ##################################################################################



##############################################   EasyOcr ###########################################
'''''Renvoie dataset ou les ad_screenshot ont ete remplce par des textes + clean'''''
"pip install easyocr torch"
from plugins.process.EasyOCRClean import add_screenshot_text

base_folder = './'
##dataset=add_screenshot_text(dataset,base_folder)
print("EasyOcr termine")

############################################  Séparer train test  ####################################
from sklearn.utils import shuffle
labels=dataset["sample_type"]
features=dataset.drop(columns=["sample_type"])
indices = shuffle(range(len(features)), random_state=1)
features = features.iloc[indices]
labels = labels.iloc[indices]
X_train,X_test,y_train,y_test=train_test_split(features,labels,random_state=42)
print("Séparation termine")

############################################    StandardScaler  #####################################
'''''Changement sur le dataset + renvoie du scaler'''''
from plugins.process.StandardScaler import sscaler

X_train_fit,X_test_fit, scaler =sscaler(X_train,X_test)
print("Standarisation termine")

############################################     OHencoder   ####################################
'''''Changement sur le dataset + renvoie de l'encodeur '''''
from plugins.process.OHEncoder import ohencoder

X_train_oh,X_test_oh, encoder=ohencoder(X_train_fit,X_test_fit)
print("One Hot encoder termine")

############################################    Word2Vec    #########################################""
'''''Renvoie dataset ou l'url a ete transforme par des vecteurs word2vec, et les vecteurs'''''
from plugins.process.Word2Vec import url_Word2vec

X_train_word2vec, word2vec_train,X_test_word2vec, word2vec_test=url_Word2vec(X_train_oh,X_test_oh)
print("Word2Vec termine")

###########################################   BERT    ###############################################
'''''Renvoie (dataset orginial + liste de 768 colonnes de bert), Dataframe 768 colonnes de bert '''''
"pip install transformers"
from plugins.process.BERT import BERT_transform

X_train_bert, bert_train= BERT_transform(X_train_word2vec)
X_test_bert, bert_test= BERT_transform(X_test_word2vec)
print("BERT termine")


#############################################     Nettoyage   ####################################
'''''''Renvoie dataset finale'''''''''
from plugins.process.Nettoyage import nettoie

X_train_fin=nettoie(X_train_bert)
X_test_fin=nettoie(X_test_bert)
y_train = y_train.replace({
    'news': 0,
    'adjusted_news':0,
    'misinfo':1
})
y_test = y_test.replace({
    'news': 0,
    'adjusted_news':0,
    'misinfo':1
})

X_train_fin.columns = X_train_fin.columns.astype(str)
X_test_fin.columns = X_test_fin.columns.astype(str)

#Sauvegarde
X_train_fin.to_csv('./Dataset/X_train.csv', index=False)
y_train.to_csv('./Dataset/y_train.csv', index=False)
X_test_fin.to_csv('./Dataset/X_test.csv', index=False)
y_test.to_csv('./Dataset/y_test.csv', index=False)
print("Processing termine")

print("\n")
print("Enregistrement des données fit dans ./Dataset/X_train,X_test... .csv")
print("\n")


X=pd.read_csv('Dataset/EasyOCR+clean.csv')
X=X.drop(columns="sample_type")

#################################################################################  Model Checking ########################################################################

#####################################""   GridSearch sur pytorch  ############################################# 
from plugins.ModelChecking.GSearchPytorch import grid_search
from plugins.ModelChecking.Pytorch import model_pytroch
from plugins.ModelChecking.TrainEvalPytorch import traineval

param_grid_pytroch = {
    "batch_size": [16, 32, 64],
    "hidden_units": [32, 64, 256 , 512],
    "learning_rate": [0.01, 0.1,0.3,1,3],
    "epochs": [500, 1000]
}
# params,acc,result= grid_search(param_grid_pytroch,X_train_fin,y_train,X_test_fin,y_test,model_pytroch,traineval)
# BATCH_SIZE = params['batch_size']
# HIDDEN_UNITS = params['hidden_units']
# LR = params['learning_rate']
# EPOCHS = params['epochs']

# print("Pytorch best model terminé: ", params)

########################################    Création Pytorch     ############################################

from plugins.ModelChecking.Pytorch import model_pytroch
# model_py_best, X_train_py, y_train_py, X_test_py, y_test_py=model_pytroch(X_train_fin,y_train,X_test_fin,y_test,BATCH_SIZE,HIDDEN_UNITS)
print("Pytorch Models terminé")

########################################    Entrainement Pytorch     ############################################
from plugins.ModelChecking.TrainEvalPytorch import traineval
# loss,accur,best=traineval(model_py_best,X_train_py, y_train_py, X_test_py, y_test_py,LR,EPOCHS)
print("Pytorch training termine")


#######################################   Sauvegarde du modéle Pytorch    ############################################
# torch.save(model_py_best.state_dict(), 'Best_Model_Pytorch.pth')

# params['acc']=round(best,2)
# with open('Pytorch_Best_Params.json', 'w') as f:
#         json.dump(params, f, indent=4)
print("\n")
print("Enregistrement du modéle Pytorch dans ./Best_Model_Pytorch.pth")
print("Enregistrement de ses meilleurs Hyperparametres et Precision (%) dans ./Pytorch_Best_Params.json")
print("\n")

########################################    Récup models     ############################################
from plugins.ModelChecking.ModelsList import models_test

models=models_test(X_train_fin.shape[1])
print("Import Models terminé")

######################################## Best Model #########################################
from plugins.ModelChecking.GridSearch import Best_Model

print("\n")
##Best_model=Best_Model(models,X_train_fin,y_train)
print("Meilleurs Hyperparametres trouvés :")
#print(Best_Model)
print("\n")
print("GridSearch terminé")


########################################    Entrainement    ############################################
from plugins.ModelChecking.Entrainement import entrainement

models_fit=entrainement(models,X_train_fin,X_test_fin,y_train,y_test)
print("Entrainement terminé")

#################################################################################  Evaluation  ################################################################################

########################################     Evaluation   ####################################
from plugins.Evaluation.Evaluation import evaluation
model_best , best_acc=evaluation(models_fit,X_test_fin,y_test)
print("Best Model: ")
print(model_best)

joblib.dump(model_best,'./Resultat/Best_Model.joblib')
print("\n")
print("Enregistrement du meilleur modele dans ./Model_GR_Best.joblib ")
print("\n")
########################################     Impotance   ####################################


from plugins.Evaluation.Correlation import correlation
Matrice_correlation=correlation(model_best,X_train_fin,y_train,X.columns.tolist())

########################################     Matrix Confusion   ####################################
from plugins.Evaluation.MatrixConfusion import matrixconfusion

# Matrice_Confusion=matrixconfusion(models,X_train_fin,y_train,X_test_fin,y_test)
print("Probleme d'indice dans Matrice de confusion sur le type de la page terminée")
#########################################################################   Plot    #########################################################

from plugins.Evaluation.plot import modele_2
# modele_2(best,best_acc )

##########################################################################    Prédiction      #################################################



########################################  Prediction  ####################################### 

#######################################       Prediction Pytorch       ######################################
from plugins.Evaluation.PytorchPrediction import prediction_pytorch

# prediction_pytorch(model_py_best,X_test_py,y_test_py,loss,accur)
# print("Pytorch Prediction termine")

