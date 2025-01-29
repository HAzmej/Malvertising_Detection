import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from BERT import BERT_transform
import easyocr
import os
import pandas as pd
import os
import time
from git import Repo
from codecarbon import EmissionsTracker 

def pull_repository(repo_url, local_path):
    """
    Clone ou met à jour un dépôt Git et retourne le chemin du sous-dossier spécifique.
    """
    # Cloner ou pull le dépôt
    if not os.path.exists(local_path):
        Repo.clone_from(repo_url, local_path)
    else:
        repo = Repo(local_path)
        repo.remotes.origin.pull()

    # Lister les sous-dossiers du dépôt cloné
    subfolders = [f.name for f in os.scandir(local_path) if f.is_dir()]

    if not subfolders:
        raise Exception("Aucun sous-dossier trouvé dans le dépôt.")

    # Retourne le chemin du premier sous-dossier trouvé
    return os.path.join(local_path, subfolders[0])

app = FastAPI()
#####je change pour l'appliquer 
class ScreenshotFolderInput(BaseModel):
    screenshots_folder: str  # Chemin vers le dossier des screenshots

@app.post("/EasyOCR_BERT/")
async def EasyOCRBERT(input_data: ScreenshotFolderInput):
    try:
        repo_url = "https://github.com/ImadBKZZ/Data.git"
        local_repo_path = "./Data"
        base_folder = pull_repository(repo_url, local_repo_path)

        #input_data = "/home/asus/Bureau/ExtensionMalvertising_2/images"

        print("IN") 
        tracker = EmissionsTracker(output_dir="./")
        tracker.start()


        #base_folder = input_data
        print(base_folder)

        if not os.path.exists(base_folder):
            return {"error": "Le dossier fourni n'existe pas."}

        
        reader = easyocr.Reader(['en', 'fr'])

        text_results = []
        for image_file in os.listdir(base_folder):
            image_path = os.path.join(base_folder, image_file)
            if os.path.isfile(image_path) and image_file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
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
        emissions = tracker.stop()
        print(f"Carbon emissions for the code bert: {emissions} kg CO2")
        return {"error": f"Erreur lors du traitement : {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=50003)
