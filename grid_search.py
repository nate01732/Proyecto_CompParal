import pandas as pd
from metodo_ensamble import cargar_y_preprocesar_datos, dividir_datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from restricciones import hiperparametros
import numpy as np
import itertools
import random
import multiprocess
import time

# Variable global para contar combinaciones
combinaciones_contadas = 0

# Función para nivelar las cargas entre procesos
def nivelacion_cargas(D, n_p):
    s = len(D) % n_p
    n_D = D[:s]
    t = int((len(D) - s) / n_p)
    out = []
    temp = []
    for i in D[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

def evaluate_set(hyperparameter_set, X_train, X_test, y_train, y_test, return_dict, lock, search_type="Grid Search"):
    global combinaciones_contadas  # Usamos la variable global
    
    total_combinations = len(hyperparameter_set)

    print(hyperparameter_set)

    for i, params in enumerate(hyperparameter_set, start=1):
        clf = MLPClassifier(random_state=42, **params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Incrementar el contador de combinaciones
        combinaciones_contadas += 1
        
        # Mostrar el progreso de las combinaciones
        print(f"{search_type} - Combinación {combinaciones_contadas}/{total_combinations} evaluada: {params}, Accuracy: {acc:.4f}")
        
        # Almacenar el resultado junto con el número de combinación
        lock.acquire()
        return_dict[combinaciones_contadas] = {**params, 'Accuracy': acc}  # Guardar los hiperparámetros con el accuracy
        lock.release()


if __name__ == '__main__':

    # Cargar y preprocesar datos
    X, y = cargar_y_preprocesar_datos('dataset/StudentPerformanceFactors.csv')

    # Dividir los datos
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    # Generación de combinaciones de hiperparámetros para cada modelo
    combinations = {}

    for model, params in hiperparametros.items():
        keys, values = zip(*params.items())  # Desempaquetamos las claves y los valores
        combinations[model] = [dict(zip(keys, v)) for v in itertools.product(*values)]  # Generamos las combinaciones

    # Mostramos el número total de combinaciones
    total_combinations = sum(len(combos) for combos in combinations.values())
    print(f'El número de combinaciones es {total_combinations}')

    # Número de hilos
    N_THREADS = 1

    # Evaluar Grid Search
    print("Iniciando Grid Search...")
    grid_splits = nivelacion_cargas(combinations, N_THREADS)
    lock = multiprocess.Lock()
    manager = multiprocess.Manager()
    return_dict_grid = manager.dict()

    start_time_grid = time.perf_counter()
    processes = []

    for i in range(N_THREADS):
        p = multiprocess.Process(target=evaluate_set,
                                 args=(grid_splits[i], X_train, X_test, y_train, y_test, return_dict_grid, lock))
        processes.append(p)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    end_time_grid = time.perf_counter()
    print(f"Grid Search finalizado en {end_time_grid - start_time_grid:.2f} segundos")


'''
    # Evaluar Random Search
    print("Iniciando Random Search...")
    random_splits = nivelacion_cargas(random_combinations, N_THREADS)
    return_dict_random = manager.dict()

    start_time_random = time.perf_counter()
    processes = []

    for i in range(N_THREADS):
        p = multiprocess.Process(target=evaluate_set,
                                 args=(random_splits[i], X_train, X_test, y_train, y_test, return_dict_random, lock))
        processes.append(p)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    end_time_random = time.perf_counter()
    print(f"Random Search finalizado en {end_time_random - start_time_random:.2f} segundos")

    # Guardar resultados en CSV para Grid Search
    df_grid = pd.DataFrame.from_dict(return_dict_grid, orient='index')
    df_grid.to_csv('resultados_grid_search_diabetes4.csv', index=False)

    # Guardar resultados en CSV para Random Search
    df_random = pd.DataFrame.from_dict(return_dict_random, orient='index')
    df_random.to_csv('resultados_random_search_diabetes4.csv', index=False)

    print("Resultados guardados en archivos CSV.")
'''