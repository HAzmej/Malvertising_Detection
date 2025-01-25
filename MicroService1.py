import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import tldextract
import numpy as np
import joblib
from joblib import load
from gensim.models import KeyedVectors
import codecarbon
from codecarbon import EmissionsTracker  
app = FastAPI()

class DatasetInput(BaseModel):
    url: str  # Chemin vers le dataset contenant les urls


@app.post("/Word2Vec/")
async def Word2Vecdef(input_data: DatasetInput):
    try:
        tracker = EmissionsTracker(output_dir="M1")
        tracker.start()
        #####   feature engineering standard scaler

        scaler = joblib.load("./Resultat/StandardScaler.joblib")



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
        print(ohencoder_df)
        ohencoder_df.to_csv("./Dataset/OHencoder_test.csv")
        
        emissions =  tracker.stop()
        print(f"Carbon emissions for the code Onehot_encoder: {emissions} kg CO2")
        
        #######     word2vec
        tracker = EmissionsTracker(output_dir="./M1")
        tracker.start()

        ###     Load modele pré entrainé recupere sur 3 millions de github
        word2vecmodel = KeyedVectors.load_word2vec_format('./Resultat/GoogleNews-vectors-negative300.bin', binary=True)

        def preprocess_url(url):
            extracted = tldextract.extract(url)
            return [extracted.domain]
        
        def get_url_embedding(url_tokens):
            """
            Calcule l'embedding moyen des tokens d'une URL.
            """
            token_embeddings = [word2vecmodel[token] for token in url_tokens if token in word2vecmodel]
            if token_embeddings:
                return np.mean(token_embeddings, axis=0)  
            else:
                return np.zeros(word2vecmodel.vector_size)


        wv = preprocess_url(input_data.url)
        print(wv)
        word2vec_var = get_url_embedding(wv)
        print(word2vec_var)
        df = pd.DataFrame([word2vec_var], columns=[f"Word2Vec_{i}" for i in range(1, 301)])

        word2vec_var_df = pd.DataFrame(word2vec_var)
        
        print(word2vec_var_df)
        ##  Sauvegarde word2vec dans .csv
        df.to_csv("./Dataset/Word2Vec_Test.csv")

        res = pd.concat([df, ohencoder_df], axis=1)

        resultat = res.to_dict(orient="records")

        emissions =  tracker.stop()
     
        print(f"Carbon emissions for the code word2vec: {emissions} kg CO2")

        return {"message": "Dataset traité avec succès", "processed_data": resultat }

    except Exception as e:
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)



