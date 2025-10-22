

from __future__ import annotations
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import os

# 1. CÁLCULO ANALÍTICO DEL GRADIENTE

x, y = sp.symbols('x y', real=True)
L_sym = (x - 2)**2 + (y + 1)**2

# Derivadas analíticas
dL_dx = sp.diff(L_sym, x)
dL_dy = sp.diff(L_sym, y)

print("Gradiente analítico:")
print("∂L/∂x =", dL_dx)
print("∂L/∂y =", dL_dy)
print()

# Funciones numéricas
L = sp.lambdify((x, y), L_sym, 'numpy')
grad_L = sp.lambdify((x, y), (dL_dx, dL_dy), 'numpy')

# 2. ALGORITMO DE GRADIENTE DESCENDENTE

def gradient_descent(alpha: float, n_iter: int = 30, start: tuple = (0, 0)) -> np.ndarray:
    xk, yk = start
    trayectoria = [(xk, yk)]

    for _ in range(n_iter):
        gx, gy = grad_L(xk, yk)
        xk = xk - alpha * gx
        yk = yk - alpha * gy
        trayectoria.append((xk, yk))

    return np.array(trayectoria)

# 3. EXPERIMENTAR CON DIFERENTES α

alphas = [0.05, 0.1, 0.3, 0.6, 1.0]
trayectorias = {a: gradient_descent(a) for a in alphas}

# 4. GRAFICAR Y GUARDAR FIGURA

def plot_trayectorias(trayectorias, filename="problema_gradiente_trayectorias.png"):
    # Crear malla de la función para contornos
    x_vals = np.linspace(-2, 4, 200)
    y_vals = np.linspace(-4, 2, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = L(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.plot(2, -1, 'r*', markersize=12, label='Mínimo analítico (2, -1)')

    for a, tray in trayectorias.items():
        plt.plot(tray[:, 0], tray[:, 1], marker='o', label=f'α = {a}')
        plt.text(tray[-1, 0], tray[-1, 1], f'{a}', fontsize=8)

    plt.title('Trayectorias del Gradiente Descendente para distintos α')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Guardar figura en la carpeta actual
    ruta_salida = os.path.join(os.getcwd(), filename)
    plt.savefig(ruta_salida, dpi=160)
    plt.close()
    print(f"✅ Figura guardada en: {ruta_salida}")

# Guardar la figura
plot_trayectorias(trayectorias)

# 5. RESULTADOS FINALES

for a, tray in trayectorias.items():
    xf, yf = tray[-1]
    Lf = L(xf, yf)
    print(f"α = {a:>4}: punto final ≈ ({xf:.4f}, {yf:.4f}), L = {Lf:.6f}")

print("\nSolución analítica: (x*, y*) = (2, -1), L* = 0")

if __name__ == "__main__":
    pass
