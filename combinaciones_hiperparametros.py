import itertools

# Espacio de búsqueda de hiperparámetros para MLP
param_grid = {
    'hidden_layer_sizes': [(50, 30), (100, 50), (128, 64, 32), (256, 128, 64)],
    'activation': ['logistic', 'tanh', 'relu', 'identity'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.00001, 0.001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [50, 100, 150]
}

# Combinaciones para el espacio de búsqueda utilizando itertools.product
keys, values = zip(*param_grid.items())
combinations_param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Espacio de búsqueda de hiperparámetros para otros modelos
hiperparametros = {
    'K-NN': {
        'n_neighbors': [1, 5, 10],  # 3 valores
        'metric': ['euclidean', 'minkowski']  # 2 valores
    },
    'Random Forest': {
        'n_estimators': [100, 200, 500],  # 3 valores
        'max_features': [3, 7, 10],  # 3 valores ajustados
        'max_depth': [5, 10],  # 2 valores ajustados
        'criterion': ['gini', 'entropy']  # 2 valores ajustados
    },
    'MLP': {
        'hidden_layer_sizes': [(100, 50, 25), (200, 100, 50)],  # 2 valores ajustados
        'activation': ['logistic', 'tanh', 'relu'],  # 3 valores ajustados
        'solver': ['sgd', 'adam'],  # 2 valores
        'learning_rate': ['constant', 'adaptive'],  # 2 valores
        'max_iter': [50, 150]  # 2 valores ajustados
    },  
}
# Combinaciones para el espacio de búsqueda de hiperparámetros de diferentes modelos
combinations_hiperparametros = {}

for model, params in hiperparametros.items():
    keys, values = zip(*params.items())
    combinations_hiperparametros[model] = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Mostrar resultados
print("Combinaciones de hiperparámetros para el MLP (param_grid):")
#for comb in combinations_param_grid:
#    print(comb)

print("\nCombinaciones de hiperparámetros para los modelos (hiperparametros):")
print(len(combinations_hiperparametros(0)))
# for model, combinations in combinations_hiperparametros.items():
#     print(f"\nModelo: {model}")
#     for comb in combinations:
#         print(comb)
