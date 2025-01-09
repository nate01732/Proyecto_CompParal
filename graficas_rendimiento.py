import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
file_path = 'resultados_grid_search_diabetes.csv'  # Reemplaza con la ruta de tu archivo
df = pd.read_csv(file_path)

# Ordenar por accuracy en orden ascendente y seleccionar los 15 mejores
mejores_resultados = df.sort_values(by="Accuracy").tail(30).reset_index()

print(mejores_resultados)

# Crear la gráfica de barras
plt.figure(figsize=(12, 8))

# Graficar los 48 primeros resultados como barras azules
plt.bar(mejores_resultados.index[:-2], mejores_resultados["Accuracy"][:-2], color="#a9e6ff", alpha=0.8)

# Graficar las últimas dos barras como rojas    
plt.bar(mejores_resultados.index[-2:], mejores_resultados["Accuracy"][-2:], color="#9987f9", alpha=0.8, label="Mejores Resultados (Grid Search)")

# Graficar el mejor resultado como verde
plt.bar(mejores_resultados.index[-3:-2], mejores_resultados["Accuracy"][-3:-2], 
        color="#072f75", hatch='/', edgecolor="#8be2fe", linewidth=2, alpha=0.8, 
        label="Mejor Resultado (Random Search y Genético)")

# Etiquetar los valores encima de cada barra
for _, row in mejores_resultados.iterrows():
    plt.text(row['index'], row['Accuracy'] + 0.002, f"{row['Accuracy']:.3f}", color="#004aad", ha="center", fontsize=20)

# Etiquetas y título
plt.xlabel("Combinaciones", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.title("Mejores resultados de cada modelo", fontsize=26)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Configurar el rango del eje Y
plt.ylim(0.7545, 0.7562)

# Modificar el tamaño de las etiquetas del eje Y
plt.tick_params(axis='y', labelsize=16)  # Cambiar el tamaño de las etiquetas del eje Y
# Modificar el tamaño de las etiquetas del eje X
plt.tick_params(axis='x', labelsize=16)  # Cambiar el tamaño de las etiquetas del eje X

# Añadir leyenda con un tamaño de fuente mayor
plt.legend(fontsize=16)  # Aquí puedes ajustar el tamaño según lo que necesites

# Ajustar márgenes
plt.tight_layout()

# Guardar la gráfica como un archivo PNG
plt.savefig('mejores_resultados_modelos.png', dpi=300)  # Puedes cambiar 'dpi' para ajustar la calidad del archivo

# Mostrar la gráfica
plt.show()
