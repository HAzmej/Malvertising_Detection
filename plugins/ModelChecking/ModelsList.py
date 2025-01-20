def models_test(input_dim):
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    model = [
        ("MLP", MLPClassifier(hidden_layer_sizes=(512,),activation='relu',max_iter=100, random_state=42)),
        ("Adaboost", AdaBoostClassifier(n_estimators=200, learning_rate=0.5)),
        ("GradientBoosting", GradientBoostingClassifier(learning_rate=0.2,max_depth=5,n_estimators=200)),
        ("RandomForest", RandomForestClassifier(criterion='gini',max_depth=10,max_features='sqrt',min_samples_split=2,n_estimators=50))]
    return model