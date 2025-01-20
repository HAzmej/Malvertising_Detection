import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import tldextract
import numpy as np
import joblib
from joblib import load

app = FastAPI()

class DatasetInput(BaseModel):
    data: str  # Chemin vers le dataset contenant les urls


@app.post("/Word2Vec/")
async def Word2Vecdef(input_data: DatasetInput):
    try:
        
        word2vecmodel = load('./Resultat/Word2Vec_model.model')
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
    
        dataset = pd.read_csv(input_data.data)

        dataset['parent_url'] = dataset['parent_url'].apply(preprocess_url)
        dataset['parent_url'] = dataset['parent_url'].apply(get_url_embedding)
        dataset["parent_url"]=dataset["parent_url"].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray))else np.nan)
        print(dataset["parent_url"])
        
        resultat = dataset["parent_url"].to_dict(orient="records")
        return {"message": "Dataset traité avec succès", "processed_data":resultat }

    except Exception as e:
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)



