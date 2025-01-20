def correlation(models,X_train,y_train,noms_features1):
    from numpy import ravel
    import numpy as np 

    import matplotlib.pyplot as plt
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
    y_list = [y_pred_MLP,y_pred_Adaboost, y_pred_GradientBoosting, y_pred_RandomForest]
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
    Model_corr = [MLP_corr,Adaboost_corr, GradientBoosting_corr, RandomForest_corr]
    
    import os
    if not os.path.exists('./plot'):
        os.makedirs('./plot')
    n = 0
    for name, model in models:
        corr_values = [corr[0] for corr in Model_corr[n]]
        Correlation = corr_values[:-(23+768)]

        Bert = Model_corr[n][-768:][0]

        first_values = Model_corr[n][:-768][0]
        onehotencoder_var = first_values[-23:]
        name_var = onehotencoder_var[:14] #name
        TLD_var = onehotencoder_var[14:23] # TLD

        mean_bert = np.mean(Bert)
        Correlation.append(mean_bert)

        mean_name_var = np.mean(name_var)
        Correlation.insert(1, mean_name_var)

        mean_TLD_var = np.mean(TLD_var)
        Correlation.insert(6, mean_TLD_var)
        
        noms_features1[3]='Word2Vec'
        noms_features1[21]='BERT'
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