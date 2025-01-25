def url_Word2vec(dataset,dataset1):
  from gensim.models import Word2Vec
  import numpy as np
  import pandas as pd
  import tldextract

  def preprocess_url(url):
    extracted = tldextract.extract(url)
    return [extracted.domain]

  # Word2Vec
  from gensim.models import KeyedVectors
  model = KeyedVectors.load_word2vec_format('./Resultat/GoogleNews-vectors-negative300.bin', binary=True)
 
  def get_url_embedding(url_tokens):
    """
    Calcule l'embedding moyen des tokens d'une URL.
    """
    token_embeddings = [model[token] for token in url_tokens if token in model]
    if token_embeddings:
      return np.mean(token_embeddings, axis=0)  # Moyenne des vecteurs
    else:
      return np.zeros(model.vector_size) 
  
  print(dataset.columns)
  dataset['parent_url'] = dataset['parent_url'].apply(preprocess_url)
  dataset1['parent_url'] = dataset1['parent_url'].apply(preprocess_url)
  print(dataset.columns)
  word2vec_var = dataset['parent_url'].apply(get_url_embedding)
  word2vec_var1 = dataset1['parent_url'].apply(get_url_embedding)
  word2vec_var=np.vstack(word2vec_var)
  word2vec_var1=np.vstack(word2vec_var1)
  encoded_train_df = pd.DataFrame(word2vec_var, index=dataset.index)
  encoded_test_df = pd.DataFrame(word2vec_var1, index=dataset1.index)

  encoded_train_df.columns = [f'Word2Vec {i}' for i in range(300)]
  encoded_test_df.columns = [f'Word2Vec {i}' for i in range(768)]
  
  
  dataset = pd.concat([dataset.drop(columns="parent_url"), encoded_train_df], axis=1)
  dataset1 = pd.concat([dataset1.drop(columns="parent_url"), encoded_test_df], axis=1)
  print(dataset.info())
  print(dataset.columns)
  return dataset, word2vec_var, dataset1, word2vec_var1