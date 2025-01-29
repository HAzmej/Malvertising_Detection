import pandas as pd
import joblib
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
        print(input_data.m3)
        ad_images=pd.DataFrame(input_data.m3)
        print("bert")
        word2vec=pd.DataFrame(input_data.m2)
        word2vec=word2vec.squeeze()
        print("url")
        word2vec_len_bert = pd.DataFrame([word2vec] * len(ad_images))
        word2vec_len_bert.index=ad_images.index

        dataset = pd.concat([word2vec_len_bert, ad_images], axis=1)
        print("concat")
        # Malvertising_Model = joblib.load("./M3/Best_Model.joblib")
        Malvertising_Model = joblib.load("/home/asus/Bureau/Malvertising_Detection/M3/Best_Model.joblib")
        print("debug")
        col=pd.read_csv("./M3/X_test.csv")
        dataset.columns=col.columns
    
        print("ok")
        y_pred=Malvertising_Model.predict(dataset)
        print("prediction")
        resultat = pd.DataFrame(y_pred, columns=['Phising'], index=dataset.index)
        print("presk fini")
        print(resultat)
        result = resultat.to_dict(orient="records")

        for row in resultat.iterrows():
            if row['phishing'] == 0:      
                return {"message": "Dataset traité avec succès", "processed_data": "-1"}
        return {"message": "Dataset traité avec succès", "processed_data": "1"}

    except Exception as e:
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=50004)

