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
 
  # def get_url_embedding(url_tokens):
  #   """
  #   Calcule l'embedding moyen des tokens d'une URL.
  #   """
  #   ListEmbed=pd.DataFrame(index=url_tokens.index)
  #   indice=ListEmbed.index
  #   z=0
  #   for tld in url_tokens:
  #     List=""
  #     n=len(tld)
  #     i=0
  #     if tld in model:
  #       token=[model[tld]]
  #     else:
  #       while i<n:
  #         List=""
  #         for j in range (i,n):
  #           List+=tld[j]
  #           if List in model:
  #             token=[model[List]]
  #             break
  #         i+=1
  #     if token is not None:
  #       ListEmbed[indice[z]] = np.mean(token, axis=0) 
  #     else:
  #       ListEmbed[indice[z]] = np.zeros(model.vector_size) 
  #     z+=1
  #   return ListEmbed
      
  def get_url_embedding(url_tokens_list):
    """
    Calcule l'embedding moyen pour une liste de tokens par URL.
    Retourne un DataFrame où chaque ligne correspond à une URL
    et contient un vecteur de dimension 300.
    
    Parameters:
    - url_tokens_list: liste de listes de tokens (chaque sous-liste représente les tokens d'une URL)
    
    Returns:
    - DataFrame (nb_urls, 300)
    """
    embeddings = []  # Liste pour stocker les vecteurs moyens des URLs
    
    for url_tokens in url_tokens_list:  # Chaque URL est une liste de tokens
        token_embeddings = []  # Liste pour stocker les embeddings des tokens d'une URL
        
        for tld in url_tokens:  # Parcourt les tokens de l'URL
            token = None
            n = len(tld)
            i = 0
            # Vérifie si le token existe directement dans le modèle
            if tld in model:
                token = model[tld]
            else:
                # Sinon, essaie de construire des sous-chaînes de gauche à droite
                while i < n:
                    sub_token = ""
                    for j in range(i, n):
                        sub_token += tld[j]
                        if sub_token in model:
                            token = model[sub_token]
                            break
                    i += 1
            
            # Si un token a été trouvé, ajoute son vecteur, sinon passe
            if token is not None:
                token_embeddings.append(token)
        
        # Calcule l'embedding moyen de l'URL ou vecteur nul si aucun token trouvé
        if token_embeddings:
            mean_embedding = np.mean(token_embeddings, axis=0)
        else:
            mean_embedding = np.zeros(model.vector_size)
        
        embeddings.append(mean_embedding)
    
    # Retourne un DataFrame avec les vecteurs d'embedding
    return pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(model.vector_size)])

  
  vv = preprocess_url(dataset['parent_url'] )

  vv1 = preprocess_url(dataset1['parent_url'])

  
  word2vec_var = get_url_embedding(vv)
  word2vec_var1 = get_url_embedding(vv1)
  print("\n")
  print(word2vec_var.shape)
  word2vec_var.index=dataset.index
  
  word2vec_var1.index=dataset1.index
  dataset = pd.concat([dataset.drop(columns="parent_url"), word2vec_var], axis=1)
  dataset1 = pd.concat([dataset1.drop(columns="parent_url"), word2vec_var1], axis=1)
  print(dataset)
  return dataset, word2vec_var, dataset1, word2vec_var1