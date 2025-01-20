def Best_Model(models,X_train,y_train):
    from numpy import ravel
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'MLP': {
            'activation':['identity', 'logistic', 'tanh', 'relu'], 
            'max_iter':[50,100,200,300], 
        },
        'Adaboost':{
            'n_estimators':[50,100,150,200],
            'learning_rate':[0.1,0.2,0.5],
        }, ## 0.5, 200 82%?
        'GradientBoosting':{
            'n_estimators':[100,200],
            'learning_rate':[0.1,0.2,0.5],
            'max_depth':[1,2,5],
        },
        'RandomForest': {
            'n_estimators': [50, 100 , 200],
            'criterion': ['gini'],
            'max_depth': [1,2,5,10],
            'min_samples_split':[2,5,10],
            'max_features':['sqrt','log2'],
        }
    }   
    best_models=[]

    # Recherche des hyperparametres
    for name, model in models:
        if name in param_grid:
            print(name)
            y_train_flattened = ravel(y_train)
            grid_search = GridSearchCV(model, param_grid[name], cv=5)
            grid_search.fit(X_train, y_train_flattened)
            print(f"Best hyperparameters for {name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_}")
            print("\n")
            best_models.append((name,grid_search.best_estimator_))
    
    import joblib
    joblib.dump(best_models,'Best_Models.joblib')
    return best_models
