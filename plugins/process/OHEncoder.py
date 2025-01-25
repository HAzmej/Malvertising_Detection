def ohencoder(X_train,X_test):
  import pandas as pd
  from sklearn.preprocessing import OneHotEncoder
  import joblib
  import tldextract

  def extract_first_level_tld(url):
    list = []
    for u in url:
      extracted = tldextract.extract(u)
      suffix_parts = extracted.suffix.split('.')
      
      if len(suffix_parts) > 1:
        list.append(suffix_parts[0])  
      else:
        list.append(extracted.suffix)
    return list
      
  encoder = OneHotEncoder(sparse=False)
  string_features=['parent_url']
  Strings = encoder.fit_transform(X_train[string_features].apply(extract_first_level_tld))
  Strings2 = encoder.transform(X_test[string_features].apply(extract_first_level_tld))
  encoded_train_df = pd.DataFrame(Strings, index=X_train.index)
  encoded_test_df = pd.DataFrame(Strings2, index=X_test.index)
    

  X_train_fit = pd.concat([X_train, encoded_train_df], axis=1)
  X_test_fit = pd.concat([X_test, encoded_test_df], axis=1)
  print(X_test_fit.columns)
  joblib.dump(encoder,'./Resultat/One_Hot_Encoder.joblib')
  print("\n")
  print("=====================================================")
  print("\n")
  print("Enregistrement du scaler dans ./Resultat/One_hot_encoder.joblib")
  print("\n")
  print("=====================================================")
  print("\n")
  return X_train_fit, X_test_fit , encoder