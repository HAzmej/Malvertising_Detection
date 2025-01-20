def Extension(dataset_path):
    print("\n")
    print("pip install numpy pandas seaborn transformers joblib sklearn torch tldextract easyocr pillow ")
    import joblib
    from joblib import load
    import pandas as pd
    import numpy as np
    import tldextract
    import time
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    a=time.time()
    ####################################################    Import de méthode   ################################################# 
    try:
        best = load('./Resultat/Model_GR_Best.joblib')
    except FileNotFoundError as fnf_error:
        print(f"Erreur : File not found : {fnf_error.filename}.")
        print("To create this file, Execute the main program : ")
        print("Use : pyhon main.py")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors du chargement des fichiers : {str(e)}")
        sys.exit(1)

    scaler = load('./Resultat/StandardScaler.joblib')
    encoder = load('./Resultat/One_hot_encoder.joblib')
    word2vecmodel = load('./Resultat/Word2Vec_model.model')

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    """"""""""""""""""""""""""""""""""  Il manque l'ajout de features   """""""""""""""""""""""""""""""""
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ####################################################    Import dataset   #################################################### 
    dataset=pd.read_csv(dataset_path)

    #####################################################  EasyOcr     ######################################################### 
    from plugins.process.EasyOCRClean import add_screenshot_text

    dataset=add_screenshot_text(dataset,'./')
    print("EASYocr finished")
    print("\n")

    ####################################################    Features / Label    ################################################# 
    label=dataset["sample_type"]
    features=dataset.drop(columns='sample_type')
    print("Division finished")
    print("\n")

    ###################################################     StandardScaler      ################################################# 
    numeric_columns = features.select_dtypes(include=['int64', 'float64']).columns
    features[numeric_columns]=scaler.transform(features[numeric_columns])
    print("Standard Scaler finished")
    print("\n")

    ###################################################    One Hot encoder     ################################################## 
    string_features=['name', 'TLD']
    Strings2 = encoder.transform(features[string_features])
    encoded_test_df = pd.DataFrame(Strings2, index=features.index)
    X_fit = pd.concat([features.drop(columns=string_features), encoded_test_df], axis=1)
    print("One Hot encoder finished")
    print("\n")

    ###################################################     BERT    ############################################################## 
    from plugins.process.BERT import BERT_transform

    X_fit, bert= BERT_transform(X_fit)
    print("BERT finished")
    print("\n")

    ##################################################     Word2Vec   ############################################################ 
    def preprocess_url(url):
        extracted = tldextract.extract(url)
        return [extracted.domain]
    def get_url_embedding(url_tokens):
      """
      Calcule l'embedding moyen des tokens d'une URL.
      """
      token_embeddings = [word2vecmodel.wv[token] for token in url_tokens if token in word2vecmodel.wv]
      if token_embeddings:
          return np.mean(token_embeddings, axis=0)  
      else:
          return np.zeros(word2vecmodel.vector_size) 
      
    X_fit['parent_url'] = X_fit['parent_url'].apply(preprocess_url)
    X_fit['parent_url'] = X_fit['parent_url'].apply(get_url_embedding)
    X_fit["parent_url"]=X_fit["parent_url"].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray))else np.nan)

    print("Word2Vec finished")

    ###################################################     Nettoyage     ########################################################## 
    from plugins.process.Nettoyage import nettoie

    X_fin=nettoie(X_fit)
    X_fin.columns = X_fin.columns.astype(str)
    label = label.replace({
        'news': 0,
        'adjusted_news':0,
        'misinfo':1
    })
    print("Nettoyage finished")
    print("\n")  

    X_fin.to_csv("Extension.csv")

    ####################################################    Prediction    ######################################################### 
    y_pred=best.predict(X_fin)
    b=time.time()
    accuracy = accuracy_score(label, y_pred)
    report = classification_report(label, y_pred)
    conf_matrix = confusion_matrix(label, y_pred)

    print(f"Evaluation results for {best}:")
    print("\n")
    print(f"Accuracy: {accuracy*100}%")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Temps d'execution:\n {round(b-a,2)} secondes ")


    #####################################################   Résultats    #####################################################
    print("Voici les prediciont:\n")
    print("Somme des prédictions y_pred=0:", sum(y_pred==0))
    print("Somme des prédictions y_pred=1:", sum(y_pred==1))
    y_pred=y_pred.astype(bool)
    y_pred_df = pd.DataFrame(y_pred, columns=['Phising'], index=features.index)
    resultat = pd.concat([features[['ad_screenshot_text']], y_pred_df], axis=1)
    resultat.to_csv("./Resultat/Resultat_Prediction.csv", index=False)
    print("\n")
    print("Les Predictions des publicités se trouvent dans ./Resultat/Resultat_Prediction.csv")

################################################################        Main        ######################################################
import sys
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("How to use : python Extension.py <chemin_vers_dataset>")
        sys.exit(1)

    folder_path = sys.argv[1]
    Extension(folder_path)


