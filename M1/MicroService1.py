import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import tldextract
import numpy as np
import joblib
from joblib import load
from gensim.models import KeyedVectors
from features import featureengineering
import codecarbon
from codecarbon import EmissionsTracker  
app = FastAPI()

class DatasetInput(BaseModel):
    url: str  


@app.post("/Word2Vec/")
async def Word2Vecdef(input_data: DatasetInput):
    try:
        tracker = EmissionsTracker(output_dir="./")
        tracker.start()
        #####   feature engineering standard scaler

        scaler = joblib.load("./M1/StandardScaler.joblib")
        vs=pd.DataFrame(input_data)
        vs = vs.rename(columns={1: "parent_url"})
        print(vs)
        ss=featureengineering(vs)
        numeric_columns = ss.select_dtypes(include=['int64', 'float64']).columns
        ss[numeric_columns]=scaler.transform(ss[numeric_columns])

        #####   One hot encoder
        def extract_first_level_tld(url):
            list = []
            
            extracted = tldextract.extract(url)
            suffix_parts = extracted.suffix.split('.')
            
            if len(suffix_parts) > 1:
                list.append(suffix_parts[0])  
            else:
                list.append(extracted.suffix)
            return list
        def manual_ohencoder(tldss):
            with open('./tld.txt', 'r') as file:
                tlds = file.readlines()

                tlds = [tld.strip() for tld in tlds]

                tlds_lower = [tld.lower() for tld in tlds]
                print("je suis la")
                
                df = pd.DataFrame([tlds_lower], columns=tlds_lower)
                for t in tldss:
                    df.loc[0] = [1 if col == t else 0 for col in df.columns]
                print("je suis pass")
                
            return df
        # Strings = encoder.fit_transform(X_train[string_features].apply(extract_first_level_tld))
        # Strings2 = encoder.transform(X_test[string_features].apply(extract_first_level_tld))
        tld = extract_first_level_tld(input_data.url)
        ohencoder_df=manual_ohencoder(tld)



        # encoder=joblib.load("./Resultat/One_Hot_Encoder.joblib")
        # print(input_data.url)
        # string = extract_first_level_tld(input_data.url)
        # print(string)
        # ohencoder_df = pd.DataFrame(string)
        # ohencoder_df=encoder.transform(ohencoder_df)
        # ohencoder_df = pd.DataFrame(ohencoder_df)
        print("j'ai finis")
        
        emissions =  tracker.stop()
        print(f"Carbon emissions for the code Onehot_encoder: {emissions} kg CO2")
        
        # #######     word2vec
        tracker = EmissionsTracker(output_dir="./")
        tracker.start()

        ###     Load modele pré entrainé recupere sur 3 millions de github
        word2vecmodel = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

        def preprocess_url(url):
            extracted = tldextract.extract(url)
            return [extracted.domain]
        
        def get_url_embedding(url_tokens_list):
            """
            Calcule l'embedding moyen pour une liste de tokens par URL.
            Retourne un DataFrame où chaque ligne correspond à une URL
            et contient un vecteur de dimension 300.
            
            Parameters:
            - url_tokens_list: liste de listes de tokens (chaque sous-liste représente les tokens d'une URL)
            
            Returns:
            - DataFrame (nb_urls, 300)
            """
            embeddings = [] 
            
            for url_tokens in url_tokens_list: 
                token_embeddings = []  
                
                for tld in url_tokens:  
                    token = None
                    n = len(tld)
                    i = 0
                  
                    if tld in word2vecmodel:
                        token = word2vecmodel[tld]
                    else:
                       
                        while i < n:
                            sub_token = ""
                            for j in range(i, n):
                                sub_token += tld[j]
                                if sub_token in word2vecmodel:
                                    token = word2vecmodel[sub_token]
                                    break
                            i += 1
                    
                  
                    if token is not None:
                        token_embeddings.append(token)
                
              
                if token_embeddings:
                    mean_embedding = np.mean(token_embeddings, axis=0)
                else:
                    mean_embedding = np.zeros(word2vecmodel.vector_size)
                
                embeddings.append(mean_embedding)
            
            
            return pd.DataFrame(embeddings, columns=[f"Word2Vec_{i}" for i in range(word2vecmodel.vector_size)])


        wv = preprocess_url(input_data.url)
        word2vec_var = get_url_embedding(wv)
        
        
        
        ##  Sauvegarde word2vec dans .csv
        word2vec_var.to_csv("./Dataset/Word2Vec_Test.csv")
        ohencoder_df.index=word2vec_var.index
        res = pd.concat([ohencoder_df, word2vec_var], axis=1)
        ss.index=res.index
        res=pd.concat([ss,res],axis=1)
        res.to_csv("../test.csv")
        resultat = res.to_dict(orient="records")

        emissions =  tracker.stop()
     
        print(f"Carbon emissions for the code word2vec: {emissions} kg CO2")

        return {"message": "Dataset traité avec succès", "processed_data": resultat }

    except Exception as e:
        emissions =  tracker.stop()
        print(f"Carbon emissions for the code Onehot_encoder: {emissions} kg CO2")
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=50002)



