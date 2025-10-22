from __future__ import annotations
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import time
import os


# 1. FUNCION, GRADIENTE Y HESSIANA

x, y = sp.symbols('x y', real=True)
f_sym = (x - 2)**2 * (y + 2)**2 + (x + 1)**2 + (y - 1)**2

# Gradiente y Hessiana analíticos
grad_f_sym = [sp.diff(f_sym, var) for var in (x, y)]
H_f_sym = sp.hessian(f_sym, (x, y))

# Convertir a funciones numéricas
f = sp.lambdify((x, y), f_sym, 'numpy')
grad_f = sp.lambdify((x, y), grad_f_sym, 'numpy')
H_f = sp.lambdify((x, y), H_f_sym, 'numpy')

print("Gradiente analítico:")
print("∂f/∂x =", grad_f_sym[0])
print("∂f/∂y =", grad_f_sym[1])
print("\nHessiana:")
sp.pprint(H_f_sym)
print("\n")


# 2. METODOS DE OPTIMIZACIÓN

def gradiente_descendente(alpha: float, start=(-2, -3), tol=1e-6, max_iter=100):
    xk, yk = start
    trayectoria = [(xk, yk)]
    for _ in range(max_iter):
        gx, gy = grad_f(xk, yk)
        x_new = xk - alpha * gx
        y_new = yk - alpha * gy
        if np.linalg.norm([x_new - xk, y_new - yk]) < tol:
            break
        xk, yk = x_new, y_new
        trayectoria.append((xk, yk))
    return np.array(trayectoria)

def newton_raphson(alpha: float, start=(-2, -3), tol=1e-6, max_iter=100):
    xk, yk = start
    trayectoria = [(xk, yk)]
    for _ in range(max_iter):
        g = np.array(grad_f(xk, yk), dtype=float)
        H = np.array(H_f(xk, yk), dtype=float)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        x_new, y_new = np.array([xk, yk]) - alpha * step
        if np.linalg.norm([x_new - xk, y_new - yk]) < tol:
            break
        xk, yk = x_new, y_new
        trayectoria.append((xk, yk))
    return np.array(trayectoria)


# 3. EXPERIMENTACION

alpha_gd = 0.01  
alpha_nr = 1.0    

start = (-2, -3)

t0 = time.time()
tray_gd = gradiente_descendente(alpha_gd, start)
t1 = time.time()
tray_nr = newton_raphson(alpha_nr, start)
t2 = time.time()

# tiempos
time_gd = t1 - t0
time_nr = t2 - t1

# resultados finales
final_gd = tray_gd[-1]
final_nr = tray_nr[-1]
f_gd = f(*final_gd)
f_nr = f(*final_nr)


# 4. VISUALIZACIÓN DE TRAYECTORIAS

def plot_trayectorias(tray_gd, tray_nr, filename="parteB_trayectorias.png"):
    x_vals = np.linspace(-3, 3, 400)
    y_vals = np.linspace(-4, 3, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)

    plt.figure(figsize=(9, 7))
    plt.contour(X, Y, Z, levels=40, cmap='viridis')
    plt.plot(start[0], start[1], 'ko', label='Punto inicial (-2, -3)')
    plt.plot(tray_gd[:,0], tray_gd[:,1], 'b-o', label=f'Gradiente Descendente (α={alpha_gd})')
    plt.plot(tray_nr[:,0], tray_nr[:,1], 'r-s', label=f'Newton-Raphson (α={alpha_nr})')
    plt.plot(final_gd[0], final_gd[1], 'bo', markersize=10)
    plt.plot(final_nr[0], final_nr[1], 'rs', markersize=10)
    plt.title('Trayectorias de convergencia: Gradiente vs Newton-Raphson')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()

plot_trayectorias(tray_gd, tray_nr)


# 5. GRÁFICO DE CONVERGENCIA DEL ERROR

def plot_convergencia(tray_gd, tray_nr, filename="parteB_convergencia.png"):
    min_x, min_y = min([*tray_gd, *tray_nr], key=lambda p: f(*p))
    f_star = f(min_x, min_y)

    err_gd = [abs(f(*p) - f_star) for p in tray_gd]
    err_nr = [abs(f(*p) - f_star) for p in tray_nr]

    plt.figure(figsize=(8,5))
    plt.semilogy(err_gd, 'b-o', label='Gradiente Descendente')
    plt.semilogy(err_nr, 'r-s', label='Newton-Raphson')
    plt.xlabel('Iteración')
    plt.ylabel('|f(x_k, y_k) - f*|')
    plt.title('Convergencia del error (escala logarítmica)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()

plot_convergencia(tray_gd, tray_nr)

# 6. COMPARATIVA NUMÉRICA
print("\n=== Resultados numéricos ===")
print(f"Gradiente Descendente: {len(tray_gd)} iteraciones, f = {f_gd:.6f}, tiempo = {time_gd:.6f} s")
print(f"Newton-Raphson:        {len(tray_nr)} iteraciones, f = {f_nr:.6f}, tiempo = {time_nr:.6f} s")




