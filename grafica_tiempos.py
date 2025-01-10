import matplotlib.pyplot as plt

# Datos en segundos
grid_search = {
    1: 10461.40,
    4: 3896.99,
    7: 3502.69,
    11: 3782.42
}

random_search = {
    1: 1515.99,
    4: 677.56,
    7: 529.46,
    11: 566.46
}

algoritmo_genetico = {
    1: 1257.04,
    4: 586.28,
    7: 489.35,
    11: 542.39
}

# Convertir a minutos
def convertir_a_minutos(datos):
    return {hilos: tiempo / 60 for hilos, tiempo in datos.items()}

algoritmo_genetico_minutos = convertir_a_minutos(algoritmo_genetico)
grid_search_minutos = convertir_a_minutos(grid_search)
random_search_minutos = convertir_a_minutos(random_search)

# Función para encontrar el mejor tiempo
def encontrar_mejor_tiempo(datos):
    mejor_hilos = min(datos, key=datos.get)
    mejor_tiempo = datos[mejor_hilos]
    return mejor_hilos, mejor_tiempo

# Encontrar mejores tiempos
mejor_genetico = encontrar_mejor_tiempo(algoritmo_genetico_minutos)
mejor_grid = encontrar_mejor_tiempo(grid_search_minutos)
mejor_random = encontrar_mejor_tiempo(random_search_minutos)

# Crear gráfica
plt.figure(figsize=(12, 8))

# Intercambiar colores: Cambiar los colores de Random Search y Algoritmo Genético
plt.plot(algoritmo_genetico_minutos.keys(), algoritmo_genetico_minutos.values(), marker='o', label="Algoritmo Genético", color='#072f75', linewidth=2)  # Ahora es el color de Random Search
plt.plot(grid_search_minutos.keys(), grid_search_minutos.values(), marker='o', label="Grid Search", color='#9987f9', linewidth=2)
plt.plot(random_search_minutos.keys(), random_search_minutos.values(), marker='o', label="Random Search", color='#8be2fe', linewidth=2)  # Ahora es el color de Algoritmo Genético

# Agregar tiempos a todos los puntos
for hilos, tiempo in algoritmo_genetico_minutos.items():
    plt.text(hilos, tiempo - 5, f"{tiempo:.1f} min", color='#072f75', fontsize=14, ha='center')  # Cambiar color para que coincida con la nueva línea

for hilos, tiempo in grid_search_minutos.items():
    plt.text(hilos, tiempo - 15, f"{tiempo:.1f} min", color='#9987f9', fontsize=14, ha='center')

for hilos, tiempo in random_search_minutos.items():
    plt.text(hilos, tiempo + 10, f"{tiempo:.1f} min", color='#8be2fe', fontsize=14, ha='center')  # Cambiar color para que coincida con la nueva línea

# Marcar puntos de mejor tiempo con un tamaño y estilo especial
plt.scatter(mejor_genetico[0], mejor_genetico[1], color='#072f75', s=100, label="Mejor Genético", edgecolor='black', zorder=5)  # Cambiar color
plt.scatter(mejor_grid[0], mejor_grid[1], color='#9987f9', s=100, label="Mejor Grid", edgecolor='black', zorder=5)
plt.scatter(mejor_random[0], mejor_random[1], color='#8be2fe', s=100, label="Mejor Random", edgecolor='black', zorder=5)  # Cambiar color

# Personalización de la gráfica
plt.title("Tiempos de ejecución por número de hilos", fontsize=26)
plt.xlabel("Número de hilos", fontsize=20)
plt.ylabel("Tiempo (minutos)", fontsize=20)
plt.xticks(list(algoritmo_genetico_minutos.keys()), fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=16, loc='upper right')  # Texto más grande para la leyenda

# Ajustar márgenes alrededor de los textos
plt.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.88)  # Ajusta según sea necesario

# Guardar como imagen (ideal para póster científico)
plt.tight_layout()
plt.savefig("tiempos_ejecucion_minutos_poster.png", dpi=300)
plt.show()
