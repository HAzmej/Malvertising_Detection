import pandas as pd
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
import time
from typing import List, Dict, Union

app = FastAPI()

class DatasetInput(BaseModel):
    m2: List[Dict[str, Union[bool, float, int]]]
    m3: List[Dict[str, float]]

    


@app.post("/predict/")
async def predict(input_data: DatasetInput):
    try:
        ad_images=input_data.m3
        word2vec=input_data.m2
        word2vec_len_bert = pd.DataFrame([word2vec.values] * len(ad_images), columns=word2vec.columns).reset_index(drop=True)

        urls = pd.concat([word2vec_len_bert, word2vec_len_bert], axis=1)

        dataset = pd.concat([urls, ad_images], axis=1)

        Malvertising_Model=load("./Best_Model.joblib")

        y_pred=Malvertising_Model.predict(dataset)
        y_pred=y_pred.astype(bool)
        y_pred_df = pd.DataFrame(y_pred, columns=['Phising'], index=dataset.index)
        resultat = pd.concat([dataset[['ad_screenshot_text']], y_pred_df], axis=1)

        for row in resultat.iterrows():
            if row['phishing'] == 0:      
                return {"message": "Dataset traité avec succès", "processed_data": "-1"}
        return {"message": "Dataset traité avec succès", "processed_data": "1"}

    except Exception as e:
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=50004)

