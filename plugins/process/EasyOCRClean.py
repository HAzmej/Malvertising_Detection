def add_screenshot_text(dataset,base_folder):

  import easyocr
  import os
  import pandas as pd
  reader = easyocr.Reader(['en', 'fr'])
  
  text_results = []
  for index, row in dataset.iterrows():
      detected_texts = []

      for col_name in ['ad_screenshot']:
          image_path = os.path.join(base_folder, row[col_name])
          if os.path.exists(image_path):  

              text = reader.readtext(image_path, detail=0)  # detail=0 retourne uniquement le texte
              detected_texts.append(' '.join(text))
              print(text)
          else:
              detected_texts.append("File not found")

      text_results.append({
          'ad_screenshot_text': detected_texts[0],
      })


  dataset['ad_screenshot_text'] = pd.DataFrame(text_results)
  dataset = dataset.drop(dataset[dataset['ad_screenshot_text'] == ""].index)
  dataset.to_csv('./Dataset/EasyOCR+clean11.csv')
  print("\n")
  print("=====================================================")
  print("\n")
  print("Enregistrement du Nouveau Dataset avec Ad_Text dans ./Dataset/EasyOCR+clean11.csv'")
  print("\n")
  print("=====================================================")
  print("\n")
  return dataset