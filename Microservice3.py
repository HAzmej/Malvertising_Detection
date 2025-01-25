import pandas as pd
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class DatasetInput(BaseModel):
    data: str  # Chemin vers le dataset contenant les urls


@app.get("/predict/")
async def Word2Vecdef(input_data: DatasetInput):
    try:
        ad_images=pd.read_csv("./Dataset/BERT.csv")
        OHencoder=pd.read_csv("./Dataset/OHencoder_test.csv")
        word2vec=pd.read_csv("./Dataset/Word2Vec_Test.csv")

        OHencoder_len_bert = pd.DataFrame([OHencoder.values] * len(ad_images), columns=OHencoder.columns).reset_index(drop=True)
        word2vec_len_bert = pd.DataFrame([word2vec.values] * len(ad_images), columns=word2vec.columns).reset_index(drop=True)

        urls = pd.concat([OHencoder_len_bert, word2vec_len_bert], axis=1)

        dataset = pd.concat([urls, ad_images], axis=1)

        Malvertising_Model=load("./Resultat/Best_Model.joblib")

        y_pred=Malvertising_Model.predict(dataset)
        y_pred=y_pred.astype(bool)
        y_pred_df = pd.DataFrame(y_pred, columns=['Phising'], index=dataset.index)
        resultat = pd.concat([dataset[['ad_screenshot_text']], y_pred_df], axis=1)

        return {"message": "Dataset traité avec succès", "processed_data":resultat }

    except Exception as e:
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)

