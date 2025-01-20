def entrainement(models,X_train,X_test, y_train, y_test):
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  import matplotlib.pyplot as plt
  import time
  from numpy import ravel
  
  for name, model in models:
      y_train_flattened = ravel(y_train)
      a=time.time()
      model.fit(X_train, y_train_flattened)
  return models


  

