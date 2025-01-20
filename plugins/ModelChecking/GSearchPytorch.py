from itertools import product
def grid_search(param_grid, X_train, y_train, X_test, y_test,model_pytroch,traineval):
    best_params = None
    best_accuracy = 0
    results = []

    # Génération des combinaisons d'hyperparamètres
    combinations = list(product(*param_grid.values()))
    keys = list(param_grid.keys())

    for combo in combinations:
        params = dict(zip(keys, combo))
        print(f"Testing params: {params}")

       
        mod, X_train_py, y_train_py, X_test_py, y_test_py = model_pytroch(
            X_train, y_train, X_test, y_test, 
            BATCH_SIZE=params["batch_size"], 
            HIDDEN_UNITS=params["hidden_units"]
        )

        # Entraînement et évaluation
        loss,acc, accuracy = traineval(mod, X_train_py, y_train_py, X_test_py, y_test_py, 
                             LR=params["learning_rate"], 
                             EPOCHS=params["epochs"])

        # Sauvegarder les résultats
        results.append((params, accuracy))
        if accuracy > best_accuracy :
            best_accuracy = accuracy
            best_params = params

    return best_params, best_accuracy, results
