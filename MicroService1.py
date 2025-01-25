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
app = FastAPI()

class DatasetInput(BaseModel):
    data: str  # Chemin vers le dataset contenant les urls


@app.post("/Word2Vec/")
async def Word2Vecdef(input_data: DatasetInput):
    try:
        from codecarbon import EmissionsTracker  
        tracker = EmissionsTracker(output_dir="./Resultat")


        tracker.start()

        dataset = pd.read_csv(input_data.data)
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
            
        encoder = OneHotEncoder(sparse=False)
        string_features=['parent_url']
        Strings = encoder.fit_transform(dataset[string_features].apply(extract_first_level_tld))
        encoded_train_df = pd.DataFrame(Strings, index=dataset.index)
        dataset = pd.concat([dataset, encoded_train_df], axis=1)

        ####### word2vec
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
    
        

        dataset['parent_url'] = dataset['parent_url'].apply(preprocess_url)
        word2vec_var = dataset['parent_url'].apply(get_url_embedding)
        word2vec_var=np.vstack(word2vec_var)
    
        encoded_train_df = pd.DataFrame(word2vec_var, index=dataset.index)
        encoded_train_df.columns = [f'Word2Vec {i}' for i in range(300)]
        dataset = pd.concat([dataset.drop(columns="parent_url"), encoded_train_df], axis=1)
    
        resultat = dataset.to_dict(orient="records")

        emissions = tracker.stop()

        print(f"Carbon emissions for the code execution1111: {emissions} kg CO2")
        return {"message": "Dataset traité avec succès", "processed_data":resultat }

    except Exception as e:
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)



