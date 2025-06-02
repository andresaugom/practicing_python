# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
# Genera dos clases linealmente separables
np.random.seed(0)
X1 = np.random.randn(50, 2) + np.array([1, 1])
X2 = np.random.randn(50, 2) + np.array([-1, -1])
X = np.vstack([X1, X2])
y = np.array([1]*50 + [0]*50)

# %%
# Definir la funcion step
def step(x):
    return 1 if x >= 0 else 0

# %%
# Inicializar pesos y sesgo
w = np.random.rand(2)
b = np.random.rand()
# Tasa de aprendizaje
eta = 1

# %%
# Fotogramas para la visualizacion
frames = []

# %%
# Prediccion del perceptron
for epoch in range(100):
    error_found = False
    for i in range(len(X)):
        y_hat = step(np.dot(w, X[i]) + b)
        if y_hat != y[i]:
            w += eta*(y[i] - y_hat)*X[i]
            b += eta*(y[i] - y_hat)
            frames.append((w.copy(), b))
            error_found = True
    if not error_found:
        break

# %%
# Animacion
fig, ax = plt.subplots()
ax.scatter(X1[:, 0], X1[:, 1], color='blue', label='Clase 1')
ax.scatter(X2[:, 0], X2[:, 1], color='red', label='Clase 0')
line, = ax.plot([], [], 'k--', lw=2, label='Frontera de decisión')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.legend()

def update(frame):
    w, b = frame
    if w[1] != 0:
        x_vals = np.array(ax.get_xlim())
        y_vals = -(w[0] * x_vals + b) / w[1]
        line.set_data(x_vals, y_vals)
    else:
        line.set_data([], [])
    return line,

ani = FuncAnimation(fig, update, frames=frames, interval=33, repeat=False)
plt.show()
# %%
