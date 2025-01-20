def ohencoder(X_train,X_test):
  import pandas as pd
  from sklearn.preprocessing import OneHotEncoder
  import joblib

  encoder = OneHotEncoder(sparse=False)
  string_features=['name', 'TLD']
  Strings = encoder.fit_transform(X_train[string_features])
  Strings2 = encoder.transform(X_test[string_features])
  encoded_train_df = pd.DataFrame(Strings, index=X_train.index)
  encoded_test_df = pd.DataFrame(Strings2, index=X_test.index)
    

  X_train_fit = pd.concat([X_train.drop(columns=string_features), encoded_train_df], axis=1)
  X_test_fit = pd.concat([X_test.drop(columns=string_features), encoded_test_df], axis=1)

  joblib.dump(encoder,'./Resultat/One_Hot_Encoder.joblib')
  print("\n")
  print("=====================================================")
  print("\n")
  print("Enregistrement du scaler dans ./Resultat/One_hot_encoder.joblib")
  print("\n")
  print("=====================================================")
  print("\n")
  return X_train_fit, X_test_fit , encoder