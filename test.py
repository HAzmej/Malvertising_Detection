import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X_train_fin=pd.read_csv('Dataset/X_train.csv')
X_test_fin=pd.read_csv('Dataset/X_test.csv')
y_train=pd.read_csv('Dataset/y_train.csv')
y_test=pd.read_csv('Dataset/y_test.csv')

X=pd.read_csv('Dataset/EasyOCR+clean.csv')
X=X.drop(columns="sample_type")

#################################################################################  Model Checking ########################################################################

#####################################""   GridSearch sur pytorch  ############################################# 
from plugins.ModelChecking.GSearchPytorch import grid_search
from plugins.ModelChecking.Pytorch import model_pytroch
from plugins.ModelChecking.TrainEvalPytorch import traineval

param_grid_pytroch = {
    "batch_size": [16, 32, 64],
    "hidden_units": [ 64,128,256,512],
    "learning_rate": [0.01, 0.1],
    "epochs": [500, 1000]
}
# params,acc,result= grid_search(param_grid_pytroch,X_train_fin,y_train,X_test_fin,y_test,model_pytroch,traineval)
BATCH_SIZE = 64 #params['batch_size']
HIDDEN_UNITS = 512 #params['hidden_units']
LR = 0.0001 #params['learning_rate']
EPOCHS = 1000 # params['epochs']

# print("Pytorch best model terminé: ", params)

########################################    Création Pytorch     ############################################

from plugins.ModelChecking.Pytorch import model_pytroch
model_py_best, X_train_py, y_train_py, X_test_py, y_test_py=model_pytroch(X_train_fin,y_train,X_test_fin,y_test,BATCH_SIZE,HIDDEN_UNITS)
print("Pytorch Models terminé")

########################################    Entrainement Pytorch     ############################################
from plugins.ModelChecking.TrainEvalPytorch import traineval
loss,accur,best=traineval(model_py_best,X_train_py, y_train_py, X_test_py, y_test_py,LR,EPOCHS)
print("Pytorch training termine")


#######################################   Sauvegarde du modéle Pytorch    ############################################
import torch
torch.save(model_py_best.state_dict(), 'Best_Model_Pytorch.pth')
import json
params['acc']=round(best,2)
with open('Pytorch_Best_Params.json', 'w') as f:
        json.dump(params, f, indent=4)

########################################    Récup models     ############################################
from plugins.ModelChecking.ModelsList import models_test
models=models_test(X_train_fin.shape[1])
print("Import Models terminé")


########################################    Entrainement    ############################################
from plugins.ModelChecking.Entrainement import entrainement

# entrainement(models,X_train_fin,X_test_fin,y_train,y_test)
print("Checking terminé")

######################################## Best Model #########################################
from plugins.ModelChecking.GridSearch import Best_Model

# Best_model=Best_Model(models,X_train_fin,y_train)
print("Best Model terminé")

#################################################################################  Evaluation  ################################################################################

########################################     Evaluation   ####################################
# import joblib
# from joblib import load
# best=load('Best_Models.joblib')
from plugins.Evaluation.Evaluation import evaluation
# model_best=evaluation(models,X_train_fin,X_test_fin,y_train,y_test)
print("Best Model: ")
# print(model_best)

########################################     Impotance   ####################################

X=X.drop(columns='ad_screenshot')
X=X.drop(columns='page_screenshot')
from plugins.Evaluation.Correlation import correlation
# Matrice_correlation=correlation(models,X_train_fin,y_train,X.columns.tolist())
print("Corrélation termine")


########################################     Matrix Confusion   ####################################
from plugins.Evaluation.MatrixConfusion import matrixconfusion

# Matrice_Confusion=matrixconfusion(models,X_train_fin,y_train,X_test_fin,y_test)
print("Matrice de confusion sur le type de la page terminée")


##########################################################################    Prédiction      #################################################



########################################  Prediction  ####################################### 
from Predict import prediction

# prediction(model_best,X_test_fin,y_test)
print("BestModel prediction termine")

#######################################       Prediction Pytorch       ######################################
from plugins.Evaluation.PytorchPrediction import prediction_pytorch

prediction_pytorch(model_py_best,X_test_py,y_test_py,loss,accur)
print("Pytorch Prediction termine")