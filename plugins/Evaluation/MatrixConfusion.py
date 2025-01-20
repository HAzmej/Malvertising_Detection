def matrixconfusion(models,X_train,y_train,X_test,y_test):
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    from numpy import ravel

    feature = 'page_type'
    X_test222=X_test
    y_test222=y_test
    X_test222.columns=X_test.columns.tolist()
    y_test222 = y_test222.reset_index(drop=True)


    diff_valeur = [0 , 1]
    True_positive = {}
    True_negative = {}
    False_positive = {}
    False_negative = {}
    for name, model in models:
        for i in diff_valeur:
            y_train_flattened = ravel(y_train)
        
            model.fit(X_train, y_train_flattened)
            
            y_pred = model.predict(X_test222)
            y_pred222=pd.DataFrame(y_pred)
            y_pred222 = y_pred222.reset_index(drop=True)
            
            mask = ((X_test222[feature]) == i)
            conf_matrix = confusion_matrix(y_test222[mask], y_pred222[mask])
            tn, fp, fn, tp = conf_matrix.ravel()
            key = f"{i}_{name}"
            True_positive[key] = tp
            True_negative[key] = tn
            False_positive[key] = fp
            False_negative[key] = fn
            
            
            # print(f"Confusion matrix for {feature} using {name} for value of {int(i)}:")
            # print(conf_matrix)
            # print("\n" + "="*50 + "\n")

    print("TP : ")
    print(True_positive)
    print("TN : ")
    print(True_negative)
    print("FP : ")
    print(False_positive)
    print("FN : ")
    print(False_negative)


    Taux_TP = {}
    Taux_TN = {}
    for k in True_positive.keys():
        Taux_TP[k] = True_positive[k]/(True_positive[k]+False_negative[k])
        Taux_TN[k] = True_negative[k]/(True_negative[k]+False_positive[k])


    print("\n")

    print("Sensibilité : ")
    print(Taux_TP)
    print("Spécificité : ")
    print(Taux_TN)

    results_df = pd.DataFrame({
        'Model_Feature': Taux_TP.keys(),
        'True Positive Rate (TPR)': Taux_TP.values(),
        'True Negative Rate (TNR)': Taux_TN.values()
    })
    print("\nRésumé des métriques :")
    print(results_df)

    
    with open('./plot/Matrix_Confusion.txt', 'w') as f:
        f.write(results_df.to_string(index=False))
