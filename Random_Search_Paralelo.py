import random
import time
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from itertools import product
from collections import Counter
import multiprocess
from restricciones import hiperparametros

random.seed(0)

# Nivelación de cargas
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

# Cargar y preprocesar datos
def cargar_y_preprocesar_datos(filepath):
    df = pd.read_csv(filepath)
    df['rounded_score'] = (df['Exam_Score'] / 10).round(0) * 10
    df['class'] = df['rounded_score'].astype(int)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    X = df.drop(['Exam_Score', 'rounded_score', 'class'], axis=1)
    y = df['class']
    return X, y

# Dividir los datos
def dividir_datos(X, y, test_size=0.2, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Función para evaluar una combinación de hiperparámetros
def evaluar_combinacion(combinacion, X_train, X_test, y_train, y_test):
    params_knn, params_rf, params_mlp = combinacion
    modelos = {
        'K-NN': KNeighborsClassifier(**params_knn),
        'Random Forest': RandomForestClassifier(random_state=0, **params_rf),
        'MLP': MLPClassifier(random_state=0, **params_mlp),
    }
    
    predicciones = {}
    for modelo, clf in modelos.items():
        clf.fit(X_train, y_train)
        predicciones[modelo] = clf.predict(X_test)
    
    # Votación
    predicciones_ensamble = []
    for i in range(len(y_test)):
        votos = [predicciones['K-NN'][i], predicciones['Random Forest'][i], predicciones['MLP'][i]]
        conteo = Counter(votos)
        max_votos = max(conteo.values())
        clases_empate = [clase for clase, cuenta in conteo.items() if cuenta == max_votos]
        prediccion_final = int(clases_empate[0] if len(clases_empate) == 1 else predicciones['MLP'][i])
        predicciones_ensamble.append(prediccion_final)
    
    accuracy_ensamble = accuracy_score(y_test, predicciones_ensamble)
    
    return combinacion, accuracy_ensamble

# Función para evaluar un subconjunto de combinaciones
def evaluate_set(combinations, X_train, X_test, y_train, y_test, return_dict, lock):
    local_results = []
    for combinacion in combinations:
        resultado = evaluar_combinacion(combinacion, X_train, X_test, y_train, y_test)
        local_results.append(resultado)

    # Guardar resultados en el diccionario compartido
    with lock:
        return_dict.extend(local_results)

# Guardar resultados en un archivo CSV
def guardar_resultados_csv(resultados, archivo_salida="resultados.csv"):
    # Crear el encabezado dinámicamente según los nombres de los hiperparámetros
    if not resultados:
        print("No hay resultados para guardar.")
        return
    
    # Obtener los nombres de los hiperparámetros
    columnas_hiperparametros = list(resultados[0][0][0].keys()) + list(resultados[0][0][1].keys()) + list(resultados[0][0][2].keys())
    
    # Crear una lista para los resultados con accuracy como última columna
    resultados_guardar = []
    for combinacion, accuracy in resultados:
        hiperparametros = list(combinacion[0].values()) + list(combinacion[1].values()) + list(combinacion[2].values())
        resultados_guardar.append(hiperparametros + [accuracy])
    
    # Escribir los resultados en un archivo CSV
    with open(archivo_salida, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columnas_hiperparametros + ['accuracy'])  # Encabezados
        writer.writerows(resultados_guardar)  # Datos
    
    print(f"Resultados guardados en {archivo_salida}")


# Función principal
def main(num_procesos=4):
    # Cargar y preprocesar datos
    X, y = cargar_y_preprocesar_datos('StudentPerformanceFactors.csv')
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    
    # Generar todas las combinaciones
    combinaciones = list(product(
        [dict(zip(hiperparametros['K-NN'].keys(), valores)) for valores in product(*hiperparametros['K-NN'].values())],
        [dict(zip(hiperparametros['Random Forest'].keys(), valores)) for valores in product(*hiperparametros['Random Forest'].values())],
        [dict(zip(hiperparametros['MLP'].keys(), valores)) for valores in product(*hiperparametros['MLP'].values())]
    ))
    # Seleccionar aleatoriamente el 15% de las combinaciones
    tamanio_muestra = int(0.15 * len(combinaciones))
    combinaciones = random.sample(combinaciones, tamanio_muestra)
      
    print(f"Total de combinaciones: {len(combinaciones)}")
    
    # Nivelar cargas
    cargas = nivelacion_cargas(combinaciones, num_procesos)
    
    # Sincronización y multiproceso
    manager = multiprocess.Manager()
    return_dict = manager.list()
    lock = manager.Lock()
    processes = []
    
    # Crear y lanzar procesos
    start_time = time.time()
    for i in range(num_procesos):
        p = multiprocess.Process(target=evaluate_set,
                                 args=(cargas[i], X_train, X_test, y_train, y_test, return_dict, lock))
        processes.append(p)
        p.start()
    
    # Esperar a que los procesos terminen
    for p in processes:
        p.join()
    
    tiempo_total = time.time() - start_time
    
    # Convertir resultados en lista
    resultados = list(return_dict)
    
    guardar_resultados_csv(resultados, archivo_salida=f"resultados_Random_Search_{num_procesos}_hilos.csv")
    print(f'El tiempo total para {num_procesos} hilos fue: {tiempo_total}')

if __name__ == "__main__":
    main(num_procesos=11)
