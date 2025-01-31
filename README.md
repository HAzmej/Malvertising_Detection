# Détection des Publicités Phishing

Ce projet a pour but de détecter les publicités malveillantes. Pour ce faire on a décidé de génerer en premier temps un modèle de Machine Learning avec un dataset qu'on a récupéré sur GitHub: 
https://github.com/eric-zeng/conpro-bad-ads-data

## 0. Pré-requis

Avant d'exécuter ce projet, il faut installer les dépendances nécessaires 
```bash
pip install -r requirement.txt
```

Récup le dossier `screenshots` du drive https://drive.google.com/file/d/12sCuxv38RPlrRKUs-hjl-JrTDQ4J_HcR/view?usp=sharing  

## 1. Génerer un modéle
```bash
python main.py
```

Le `main.py` génere le meilleur modéle entrainé sur le dataset trouvé en vérifiant:
  - L'optimisation des **hyperparamètres**.
  - La **Précision** du modèle.
  - L'analyse énergétique avec **CodeCarbon**.
  - L'**importance** des features.

Les résultats de cette analyse sont enregistrés dans le dossier `Resultat`: 
  - `Model_GR_Best.joblib` : Le **Meilleur** modèle entraîné.
  - `StandardScaler.joblib` : Le scaler utilisé pour normaliser les données.
  - `One_Hot_Encoder.joblib` : L'encodeur des variables catégoriques.
  - `Word2Vec_model.model` : Le modèle Word2Vec utilisé pour encoder les URLs.

Remarques:

1. Les résultats des test sont stockés dans le dossier `plot` sous le nom :
  `{nom_model}_{%precision}.txt`

2. Les hyperparametres optimaux pour chaque model sont enregistrés dans :
  `Best_Models.joblib`

![Figure : Architecture de Train/Test](./img/testtrain.png)

## 2. Prédiction

```bash
python Extension.py <chemin_vers_dataset>
```

Afin de prédire si une publicité est malveillante, le modéle a besoin de deux inputs:
  - Les URLs.
  - Les images des publicités.
    
### 2.1 MicroService 1 : Feature Engineering + One-Hot Encoder + Word2Vec

Encodage des Urls 
- Lancer le serveur sur l'adresse `http://127.0.0.1:50002` :
```bash
python MicroService1.py
```

- API Post:
```bash
curl -X POST "http://127.0.0.1:50002/Word2Vec/" \ 
-H "Content-Type: application/json" \ 
-d '{ "url": "URL"
}'
```


### 2.2 MicroService 2 : EasyOCR + BERT

Transforme les images en valeurs numériques acceptées par le modéle

  - EasyOCR : pour extraire le texte des images.
  - BERT : pour encoder les textes extraits.

Pour ce faire: 
  - Lancer le serveur sur l'adresse `http://127.0.0.1:50003` :
```bash
python MicroService2.py
```

  - API Post:
```bash
curl -X POST "http://127.0.0.1:50003/EasyOCR+BERT/" \ 
-H "Content-Type: application/json" \ 
-d '{ "screenshots_folder": <chemin_vers_dossier_screesnhots>
}'
```

### 2.3 MicroService 3 : Prédiction

Récupére les 2 sorties des Microservices 1 et 2 et prédit le caractére de la publicités

Le résultat est stocké dans `Resultat/Resultat_Prediction`

![Figure : Architecture de la prédiction](./img/predict.png)

# Contact

Questions ou Suggestions, n'hésitez pas à me contacter à :

📧 mejri@insa-toulouse.fr

📧 mejri.hazem2070@gmail.com




