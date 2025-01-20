from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from plugins.process.EasyOCRClean import add_screenshot_text
from transformers import pipeline
from plugins.process.BERT import BERT_transform

app = FastAPI()

# Définir le modèle de données attendu
class DatasetInput(BaseModel):
    data: str  # Chemin vers le fichier CSV contenant le dataset
    screenshots_folder: str  # Chemin vers les screenshots

# Initialiser le pipeline BERT pour l'analyse
bert_pipeline = pipeline("text-classification", model="bert-base-uncased", return_all_scores=True)

@app.post("/EasyOCR/")
async def EasyOCR(input_data: DatasetInput):
    try:
        # Charger le dataset à partir du chemin CSV
        dataset = pd.read_csv(input_data.data)

        # Appeler la fonction EasyOCRClean pour traiter le dataset
        updated_dataset = add_screenshot_text(dataset, input_data.screenshots_folder)
        updated_dataset = updated_dataset.drop(updated_dataset[updated_dataset['ad_screenshot_text'] == ""].index)

        # Ajouter une colonne pour les prédictions BERT
        updated_dataset, bert = BERT_transform(updated_dataset)

        # Convertir le DataFrame en JSON pour le retour
        result = updated_dataset.to_dict(orient="records")
        return {"message": "Dataset traité avec succès", "processed_data": result}

    except Exception as e:
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
