import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import tldextract
import numpy as np
import joblib
from joblib import load
from gensim.models import KeyedVectors
from sklearn.preprocessing import OneHotEncoder
import codecarbon
app = FastAPI()

class DatasetInput(BaseModel):
    data: str  # Chemin vers le dataset contenant les urls


@app.post("/Word2Vec/")
async def Word2Vecdef(input_data: DatasetInput):
    try:
        from codecarbon import EmissionsTracker  
        tracker = EmissionsTracker(output_dir="./M1")
        
        tracker.start()
        dataset=pd.read_csv(input_data.data)
        # dataset = pd.read_csv("./Dataset/urls.csv")
        #####feature engineering standard scaler
     




        #####One hot encoder
        def extract_first_level_tld(url):
            list = []
            for u in url:
                extracted = tldextract.extract(u)
                suffix_parts = extracted.suffix.split('.')
                
                if len(suffix_parts) > 1:
                    list.append(suffix_parts[0])  
                else:
                    list.append(extracted.suffix)
            return list
       
        encoder=joblib.load("./Resultat/One_Hot_Encoder.joblib")
    
        string_features=['parent_url']
        Strings = encoder.transform(dataset[string_features].apply(extract_first_level_tld))
        ohencoder_df = pd.DataFrame(Strings, index=dataset.index)
        ohencoder_df.to_csv("./Dataset/OHencoder_test.csv")
        
        emissions =  tracker.stop()
        print(f"Carbon emissions for the code Onehot_encoder: {emissions} kg CO2")
        ####### word2vec
        tracker = EmissionsTracker(output_dir="./M1")


        tracker.start()
        ###Load modele pré entrainé recupere sur 3 millions de github
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
    
        

        dataset["parent_url"] = dataset["parent_url"].apply(preprocess_url)
        word2vec_var = dataset["parent_url"].apply(get_url_embedding)
        word2vec_var=np.vstack(word2vec_var)

        word2vec_var_df = pd.DataFrame(word2vec_var, index=dataset.index)
        word2vec_var_df.columns = [f'Word2Vec {i}' for i in range(300)]

        ## Sauvegarde word2vec dans .csv
        word2vec_var_df.to_csv("./Dataset/Word2Vec_Test.csv")

        res = pd.concat([word2vec_var_df, ohencoder_df], axis=1)

        resultat = res.to_dict(orient="records")

        emissions =  tracker.stop()
        print(f"Carbon emissions for the code word2vec: {emissions} kg CO2")

        return {"message": "Dataset traité avec succès", "processed_data":resultat }

    except Exception as e:
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)



