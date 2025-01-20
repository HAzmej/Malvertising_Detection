import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from plugins.process.EasyOCRClean import add_screenshot_text
from plugins.process.BERT import BERT_transform

app = FastAPI()


class DatasetInput(BaseModel):
    data: str  # Chemin vers le dataset
    screenshots_folder: str  # Chemin vers les screenshots

@app.post("/EasyOCR+BERT/")
async def EasyOCRBERT(input_data: DatasetInput):
    try:
       
        dataset = pd.read_csv(input_data.data)

       
        updated_dataset = add_screenshot_text(dataset, input_data.screenshots_folder)
        updated_dataset = updated_dataset.drop(updated_dataset[updated_dataset['ad_screenshot_text'] == ""].index)

       
        updated_dataset, bert = BERT_transform(updated_dataset)

        # Convertir le DataFrame en JSON pour le retour
        result = updated_dataset.to_dict(orient="records")
        return {"message": "Dataset traité avec succès", "processed_data": result}

    except Exception as e:
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
