# Détection des Publicités Phishing

Ce projet a pour but de détecter les publicités malveillantes. Pour ce faire on a décidé de génerer en premier temps un Modele de Machine Learning avec un dataset qu'on a récupéré sur GitHub: 
https://github.com/eric-zeng/conpro-bad-ads-data

## 1. Génerer un modéle

[CMD] python main.py

Le main.py génere le meilleur modéle entrainé sur le dataset trouvé en vérifiant:
  - Les hyperparametres
  - La Précision
  - Le CodeCarbon
  - L'influence des features

Les résultats de cette analyse sont enregistrés dans le dossier **Resultat**: 
  - Model_GR_Best.joblib
  - StandardScaler.joblib
  - One_Hot_Encoder.joblib
  - Word2Vec_model.model

NB: Les résultats des test sont stockés dans le dossier plot sous le nom de **{nom_model}_{%precision}.txt**

NB2: Les hyperparametres optimaux pour chaque model sont enregistrés dans **Best_Models.joblib**

## 2. Prédiction
[CMD] python Extension.py <chemin_vers_dataset>

Afin de prédire si une publicité est malveillante, le modéle à besoin de deux inputs:
  - Les URLs
  - Les images des publicités
    
### 2.1 MicroService 1 : Word2Vec

Encodage des Urls 

### 2.2 MicroService 2 : EasyOCR + BERT

Transforme les images en valeurs numériques acceptés par le modéle

Pour ce faire: 
  - Executer le fichier python MicroService.py
  - [CMD] curl -X POST "http://127.0.0.1:8000/EasyOCR/" \ -H "Content-Type: application/json" \ -d '{
  "data": "./Dataset/NewDataset.csv",
  "screenshots_folder": "./"
}'

### 2.3 MicroService 3 : Prédiction

Récupére les 2 sorties des Microservices 1 et 2 et prédit le caractére de la publicités; le résultat est stocké dans **Resultat/Resultat_prediction**
