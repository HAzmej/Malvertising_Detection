import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from BERT import BERT_transform
import easyocr
import os
import pandas as pd

app = FastAPI()
#####je change pour l'appliquer 
class ScreenshotFolderInput(BaseModel):
    screenshots_folder: str  # Chemin vers le dossier des screenshots

@app.post("/EasyOCR+BERT/")
async def EasyOCRBERT(input_data: ScreenshotFolderInput):
    try:
        from codecarbon import EmissionsTracker  
        tracker = EmissionsTracker(output_dir="./")
        tracker.start()
        base_folder = input_data.screenshots_folder

        if not os.path.exists(base_folder):
            return {"error": "Le dossier fourni n'existe pas."}

        
        reader = easyocr.Reader(['en', 'fr'])

        text_results = []
        for image_file in os.listdir(base_folder):
            image_path = os.path.join(base_folder, image_file)

            
            if os.path.isfile(image_path) and image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    text = reader.readtext(image_path, detail=0)  # detail=0 retourne uniquement le texte
                    detected_text = ' '.join(text)
                    print(f"Texte détecté pour {image_file}: {detected_text}")
                except Exception as e:
                    detected_text = f"Erreur lors du traitement de {image_file}: {str(e)}"

                text_results.append({
                    'ad_screenshot_text': detected_text
                })

        ad_screenshot_text= pd.DataFrame(text_results)
        ad_screenshot_text = ad_screenshot_text.drop(ad_screenshot_text[ad_screenshot_text['ad_screenshot_text'] == ""].index)
        emissions = tracker.stop()

        print(f"Carbon emissions for the code easyocr: {emissions} kg CO2")

        tracker = EmissionsTracker(output_dir="./")
        tracker.start()
        ### BERT
        ad_screenshot_text, bert = BERT_transform(ad_screenshot_text)

        # Convertir le DataFrame en JSON pour le retour
        
        ad_screenshot_text.to_csv("./Dataset/BERT.csv")
        result = ad_screenshot_text.to_dict(orient="records")

        emissions = tracker.stop()

        print(f"Carbon emissions for the code bert: {emissions} kg CO2")
        return {"message": "Dataset traité avec succès", "processed_data": result}

    except Exception as e:
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=50003)
