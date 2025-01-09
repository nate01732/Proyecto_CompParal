"""" Configuraciones de Hiperparámetros """
hiperparametros_ = {
    'K-NN': {
        'n_neighbors': list(range(1, 11, 2)),
        'metric': ['euclidean', 'minkowski', 'manhattan']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_features': [3, 5, 7, 10],
        'max_depth': [3, 10, 20],
        'criterion': ['gini', 'entropy', 'log_loss']
    },
    'MLP': {
        'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [50, 100, 150]
    },
}

hiperparametros = {
    'K-NN': {
        'n_neighbors': [1, 5, 10],  # 5 valores
        'metric': ['euclidean']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 500],  # 3 valores
        'max_features': [3, 7, 10],  # 3 valores ajustados
        'max_depth': [5, 10],  # 2 valores ajustados
        'criterion': ['gini']  # 2 valores ajustados
    },
    'MLP': {
        'hidden_layer_sizes': [(100, 50, 25), (200, 100, 50)],  # 2 valores ajustados
        'activation': ['logistic', 'tanh','relu'],
        'solver': ['adam'],  # 2 valores
        'learning_rate': ['constant', 'adaptive'],  # 2 valores
        'max_iter': [100, 200]  # 2 valores ajustados
    },
}
