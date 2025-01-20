def BERT_transform(dataset):
  from transformers import BertTokenizer, BertModel
  import torch
  import pandas as pd
  import numpy as np

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')

  def encode_text_column(text_column):
      embeddings = []
      for text in text_column:
          inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
          outputs = model(**inputs)
          # Extraire le vecteur CLS (premier token)
          cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
          embeddings.append(cls_embedding)
      return embeddings

  Bert = encode_text_column(dataset['ad_screenshot_text'])
  Bert=np.vstack(Bert)
  
  encoded_bert = pd.DataFrame(Bert, index=dataset.index)
  encoded_bert.columns = [f'Screenshot_BERT {i}' for i in range(768)]
  dataset = pd.concat([dataset.drop(columns='ad_screenshot_text'), encoded_bert], axis=1)
  
  return dataset , encoded_bert