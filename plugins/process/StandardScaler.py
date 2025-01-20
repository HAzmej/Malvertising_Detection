def sscaler(X_train,X_test):
  from sklearn.preprocessing import StandardScaler
  import joblib
  ss=StandardScaler()
  numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
  X_train[numeric_columns] = ss.fit_transform(X_train[numeric_columns])
  X_test[numeric_columns]=ss.transform(X_test[numeric_columns])
  joblib.dump(ss,'./Resultat/StandardScaler.joblib')
  print("\n")
  print("=====================================================")
  print("\n")
  print("Enregistrement du scaler dans ./Resultat/StandardScaler.joblib")
  print("\n")
  print("=====================================================")
  print("\n")
  return X_train, X_test, ss