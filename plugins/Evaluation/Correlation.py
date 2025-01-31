def correlation(models,X_train,y_train,noms_features1):
    from numpy import ravel
    import numpy as np 
   
    # models={
    #     ('GradientBoosting',models)
    # }

    import matplotlib.pyplot as plt
    noms_features1[0]='Word2Vec'
    noms_features1.append('One-hot encoder')
    noms_features1.append('BERT')
    for name, model in models:
        y_train_flattened = ravel(y_train)

        model.fit(X_train, y_train_flattened)
        if name == "MLP":
            y_pred_MLP = model.predict(X_train)
        if name == "Adaboost":
            y_pred_Adaboost = model.predict(X_train)
        if name == "GradientBoosting":
            y_pred_GradientBoosting = model.predict(X_train)
        if name == "RandomForest":
            y_pred_RandomForest = model.predict(X_train)
    

    noms_features=X_train.columns.tolist()
    from scipy.stats import pearsonr
    y_list = [y_pred_MLP,y_pred_Adaboost,y_pred_GradientBoosting,y_pred_RandomForest]
    # y_list = [y_pred_GradientBoosting]
    n = 0
    MLP_corr = []
    Adaboost_corr = []
    GradientBoosting_corr = []
    RandomForest_corr = []

    for y in y_list:
        for feat in noms_features:
            if feat in X_train.columns:
                correlation_coefficient = pearsonr(X_train[feat], y)
                # print(feat + " : " + str(correlation_coefficient))
                if n == 0:
                    MLP_corr.append(correlation_coefficient)
                if n == 1:
                    Adaboost_corr.append(correlation_coefficient)
                if n == 2:
                    GradientBoosting_corr.append(correlation_coefficient)
                if n == 3:
                    RandomForest_corr.append(correlation_coefficient)
        print("\n")
        n +=1
    # Model_corr = [MLP_corr,Adaboost_corr, GradientBoosting_corr, RandomForest_corr]
    Model_corr = [MLP_corr,Adaboost_corr,GradientBoosting_corr,RandomForest_corr]
    
    import os
    if not os.path.exists('./plot'):
        os.makedirs('./plot')
    n = 0
    for name, model in models:
        corr_values = [corr[0] for corr in Model_corr[n]]
        Correlation = corr_values[:-(300+464+768)]
        print(len(Correlation))
        Bert = Model_corr[n][-768:][0]

        first_values = Model_corr[n][:-768][0]
        
        onehotencoder_var = first_values[-(300+464):]

        tld = onehotencoder_var[:464] #tld
        word2v = onehotencoder_var[464:300] # url

        mean_tld = np.mean(tld)
        Correlation.append(mean_tld)
        
        mean_bert = np.mean(Bert)
        Correlation.append(mean_bert)
        
        mean_word2v=np.mean(word2v)
        Correlation.insert(0, mean_word2v)

        # print(noms_features1)

        plt.figure(figsize=(10, 6))
        plt.bar(noms_features1, Correlation, color='blue')
        plt.xlabel('Features')
        plt.ylabel('Coefficient de Corrélation')
        plt.title('Coefficients de Corrélation entre les features et la prediction du model ' + name )
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(f'./plot/Corrélation_{name}.png')  
        plt.close()  

        # plt.show()
        n += 1
    print("Corrélation Sauvegarder dans ./plot/Correlation_[Modele].png")
    import pandas as pd
    corr_matrix = pd.DataFrame(Model_corr, columns=noms_features)
    corr_matrix.to_csv('./Dataset/Correlation_Matrix.csv', index=False)    
    print("Corrélation Sauvegarder dans ./Dataset/Correlation_Matrix.png")
    return Model_corr