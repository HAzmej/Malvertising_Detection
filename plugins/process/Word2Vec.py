def url_Word2vec(dataset,dataset1):
  import re
  from gensim.models import Word2Vec
  import numpy as np

  import tldextract

  def preprocess_url(url):
    extracted = tldextract.extract(url)
    return [extracted.domain]


  dataset['parent_url'] = dataset['parent_url'].apply(preprocess_url)
  sentences = dataset['parent_url'].tolist()
  
  # Word2Vec
  model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
  model.save('./Resultat/Word2Vec_model.model')
  print("\n")
  print("=====================================================")
  print("\n")
  print("Enregistrement du modele dans ./Resultat/Word2Vec_model.model")
  print("\n")
  print("=====================================================")
  print("\n")
  # Obtenir des embeddings pour les tokens
  def get_url_embedding(url_tokens):
      """
      Calcule l'embedding moyen des tokens d'une URL.
      """
      token_embeddings = [model.wv[token] for token in url_tokens if token in model.wv]
      if token_embeddings:
          return np.mean(token_embeddings, axis=0)  # Moyenne des vecteurs
      else:
          return np.zeros(model.vector_size) 
  
  dataset['parent_url'] = dataset['parent_url'].apply(get_url_embedding)
  word2vec_var=dataset['parent_url']
  dataset["parent_url"]=dataset["parent_url"].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray))else np.nan)
  
  dataset1['parent_url'] = dataset1['parent_url'].apply(preprocess_url)
  dataset1['parent_url'] = dataset1['parent_url'].apply(get_url_embedding)
  word2vec_var1=dataset1['parent_url']
  dataset1["parent_url"]=dataset1["parent_url"].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray))else np.nan)

  

  return dataset, word2vec_var, dataset1, word2vec_var1