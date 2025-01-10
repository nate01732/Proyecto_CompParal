import matplotlib.pyplot as plt

# Datos en segundos proporcionados por el usuario
grid_search = {
    1: 1257.04,
    4: 586.28,
    7: 489.35,
    11: 542.39
}

# Procesadores y tiempos
procesadores = list(grid_search.keys())
tiempos = list(grid_search.values())

# Cálculo de speed-up
speedup = [tiempos[0] / tiempo for tiempo in tiempos]

# Cálculo de eficiencia
eficiencia = [s / p for s, p in zip(speedup, procesadores)]

# Crear figura y ejes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar speed-up (eje izquierdo)
color = 'tab:blue'
ax1.set_xlabel('Cores', fontsize=16)
ax1.set_ylabel('Speedup', color=color, fontsize=16)
ax1.plot(procesadores, speedup, 'o-', color=color, label='Speedup', markersize=8, linewidth=1.5)
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.set_ylim(0, max(speedup) + 1)

# Crear segundo eje para la eficiencia
ax2 = ax1.twinx()  # Comparte el eje x
color = 'tab:red'
ax2.set_ylabel('Eficiencia', color=color, fontsize=16)
ax2.plot(procesadores, eficiencia, 'd-', color=color, label='Eficiencia', markersize=8, linewidth=1.5)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
ax2.set_ylim(0, 1.1)

# Título y leyendas
plt.title('Speedup y Eficiencia Algoritmo Genético', fontsize=20)
fig.tight_layout()

# Mostrar gráfica
plt.show()
