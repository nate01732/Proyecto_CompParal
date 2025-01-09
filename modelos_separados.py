import random
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from restricciones import hiperparametros

random.seed(0)
random_seed = 0

# 1. Cargar y Preprocesar Datos
def cargar_y_preprocesar_datos(filepath):
    df = pd.read_csv(filepath)
    
    """ Este conjunto de datos es originalmente para una regresión, debido que
        el objetivo es hacer una predicción de un valor continuo como lo son las 
        calficaciones.
        Sin embargo lo que realizamos es redondear las calificaciones (0,10,20,...,100),
        para poder tomarlas como clases y así utilizar modelos de clasificación
    """
    
    # Redondear las calificaciones a los múltiplos de 10 más cercanos
    df['rounded_score'] = (df['Exam_Score'] / 10).round(0) * 10
    df['class'] = df['rounded_score'].astype(int)
    
    # Convertir columnas categóricas en variables númericas
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Separar características y etiqueta
    X = df.drop(['Exam_Score', 'rounded_score', 'class'], axis=1)
    y = df['class']
    
    return X, y

# 2. Dividir los Datos (Entrenamiento/Testeo)
def dividir_datos(X, y, test_size=0.2, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

""" Este programa es solo para hacer una prueba de una configuración de hiperparámetros"""
# 3. Función para seleccionar hiperparámetros aleatorios
def seleccionar_hiperparametros(hiperparametros):
    return {param: random.choice(valores) for param, valores in hiperparametros.items()}

# 4. Función para Entrenar y Evaluar Modelos
def entrenar_y_evaluar_modelos(X_train, X_test, y_train, y_test, hiperparametros):
    resultados = {}
    start_time = time.time()

    for modelo, params in hiperparametros.items():
        params_seleccionados = seleccionar_hiperparametros(params)
        
        # Inicializar el modelo correspondiente
        if modelo == 'K-NN':
            clf = KNeighborsClassifier(**params_seleccionados)
        elif modelo == 'Random Forest':
            clf = RandomForestClassifier(random_state=random_seed, **params_seleccionados) 
        elif modelo == 'MLP':
            clf = MLPClassifier(random_state=random_seed, **params_seleccionados)  
            
        # Entrenar el modelo
        clf.fit(X_train, y_train)
        
        # Predecir y calcular el accuracy
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Almacenar los resultados
        resultados[modelo] = {
            'hiperparametros': params_seleccionados,
            'predicciones': y_pred,
            'clase_correcta': y_test.values,
            'exactitud': accuracy
        }

    tiempo = time.time() - start_time
    return resultados, tiempo

# 5. Función para Mostrar Resultados
def mostrar_resultados(resultados, tiempo):
    for modelo, res in resultados.items():
        print(f"\nModelo: {modelo}")
        print(f"Predicciones: {res['predicciones'][:10]} (primeras 10)")
        print(f"Clase Correcta: {res['clase_correcta']}")
        print(f"Exactitud: {res['exactitud']:.8f}")
    print(f"Tiempo de ejecución: {tiempo:.4f} segundos")

#  Función main
def main():
    # Cargar y preprocesar datos
    X, y = cargar_y_preprocesar_datos('StudentPerformanceFactors.csv')
    
    # Dividir los datos
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    
    # Entrenar y evaluar modelos
    resultados, tiempo = entrenar_y_evaluar_modelos(X_train, X_test, y_train, y_test, hiperparametros)
    
    # Mostrar resultados
    mostrar_resultados(resultados, tiempo)

if __name__ == "__main__":
    main()
