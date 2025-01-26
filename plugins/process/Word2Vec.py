def url_Word2vec(dataset,dataset1):
  from gensim.models import Word2Vec
  import numpy as np
  import pandas as pd
  import tldextract

  def preprocess_url(url):
    import re
    list=[]
        # Ajoutez 'www.' si le domaine ne commence pas par 'www.'
    for u in url:
      if not re.match(r'^https?://(www\.)?', u ):
        url = re.sub(r'^(https?://)', r'\1www.', u)
      extracted = tldextract.extract(u)
      list.append([extracted.domain])
    return list
  # Word2Vec
  from gensim.models import KeyedVectors
  model = KeyedVectors.load_word2vec_format('./Resultat/GoogleNews-vectors-negative300.bin', binary=True )
 
  def get_url_embedding(url_tokens):
    """
    Calcule l'embedding moyen des tokens d'une URL.
    """
    token_embeddings = [model[url_tokens] for token in url_tokens if token in model]
    if token_embeddings:
      return np.mean(token_embeddings, axis=0) 
    else:
      return np.zeros(model.vector_size) 
  
  print(dataset.head())
  vv = preprocess_url(dataset['parent_url'] )
  print(vv)
  print("\n")
  vv1 = preprocess_url(dataset1['parent_url'])
  print(vv1)
  print(dataset.columns)
  word2vec_var = dataset['parent_url'].apply(get_url_embedding)
  word2vec_var1 = dataset1['parent_url'].apply(get_url_embedding)
  word2vec_var=np.vstack(word2vec_var)
  word2vec_var1=np.vstack(word2vec_var1)
  encoded_train_df = pd.DataFrame(word2vec_var, index=dataset.index)
  encoded_test_df = pd.DataFrame(word2vec_var1, index=dataset1.index)

  encoded_train_df.columns = [f'Word2Vec {i}' for i in range(300)]
  encoded_test_df.columns = [f'Word2Vec {i}' for i in range(300)]
  
  
  dataset = pd.concat([dataset.drop(columns="parent_url"), encoded_train_df], axis=1)
  dataset1 = pd.concat([dataset1.drop(columns="parent_url"), encoded_test_df], axis=1)
  print(dataset.info())
  print(dataset.columns)
  return dataset, word2vec_var, dataset1, word2vec_var1