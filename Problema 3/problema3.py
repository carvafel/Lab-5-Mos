import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # activa proyección 3D
import csv

#Definición simbólica (∇f, ∇²f) 
#Variables simbólicas
x, y = sp.symbols('x y', real=True)

#Función de Rosenbrock desplazada
f_sym = (x - 1)**2 + 100*(y - x**2)**2

#Gradiente (vector columna) y Hessiana (matriz 2x2)
grad_sym = sp.Matrix([sp.diff(f_sym, x), sp.diff(f_sym, y)])
hess_sym = sp.hessian(f_sym, (x, y))

#Conversión a funciones numéricas (para evaluación vectorizada con NumPy)
f    = sp.lambdify((x, y), f_sym,      'numpy')
grad = sp.lambdify((x, y), grad_sym,   'numpy')
hess = sp.lambdify((x, y), hess_sym,   'numpy')

#Newton–Raphson en R^2 
def newton2d(f, grad, hess, x0, y0, alpha=0.6, tol=1e-10, max_iter=300):
    """
    Minimiza f(x,y) resolviendo ∇f=0 con Newton amortiguado:

        [x_{k+1}, y_{k+1}]^T = [x_k, y_k]^T - alpha * (∇²f(x_k,y_k))^{-1} ∇f(x_k,y_k)
    """
    xk, yk = float(x0), float(y0)

    #Historial para análisis y gráficas
    xs, ys, fs, gns = [xk], [yk], [float(f(xk, yk))], []

    for k in range(max_iter):
        #∇f y ∇²f en el punto actual
        gk = np.array(grad(xk, yk), dtype=float).reshape(2, 1)
        Hk = np.array(hess(xk, yk), dtype=float)

        #Norma del gradiente 
        gn = float(np.linalg.norm(gk, 2))
        gns.append(gn)
        if gn < tol:
            break

        #Dirección de Newton: resolver Hk * p = gk 
        try:
            pk = np.linalg.solve(Hk, gk)
        except np.linalg.LinAlgError:
            #Hessiana singular 
            print(f"Advertencia: Hessiana singular o mal condicionada en la iteración {k}.")
            break

        #Paso amortiguado
        xk = xk - float(alpha * pk[0, 0])
        yk = yk - float(alpha * pk[1, 0])

        #Guardar en el historial
        xs.append(xk)
        ys.append(yk)
        fs.append(float(f(xk, yk)))

    return {"x": xs, "y": ys, "f": fs, "grad_norm": gns, "iters": len(xs) - 1}

#Ejecutar Newton desde (0,10)
hist = newton2d(f, grad, hess, x0=0.0, y0=10.0, alpha=0.6, tol=1e-10, max_iter=300)

#Guardar log de iteraciones a CSV 
with open('problema3a_iteraciones.csv', 'w', newline='', encoding='utf-8') as fp:
    w = csv.writer(fp)
    w.writerow(['k', 'x_k', 'y_k', 'f(x_k,y_k)', '||grad||_2'])
    for k in range(len(hist['x'])):
        gk = np.array(grad(hist['x'][k], hist['y'][k]), dtype=float).reshape(2, 1)
        w.writerow([k, hist['x'][k], hist['y'][k], hist['f'][k], float(np.linalg.norm(gk, 2))])

print("Iteraciones realizadas:", hist['iters'])
print("Último punto:", (hist['x'][-1], hist['y'][-1]))
print("f(último):", hist['f'][-1])

#Gráfica 3D de la superficie con ruta
def plot_superficie_con_ruta(hist, xlim=(-2, 2), ylim=(-1, 11),
                             nx=150, ny=150, filename='problema3a_superficie_ruta.png'):
    #Malla para la superficie
    X = np.linspace(*xlim, nx)
    Y = np.linspace(*ylim, ny)
    XX, YY = np.meshgrid(X, Y)
    ZZ = f(XX, YY)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XX, YY, ZZ, alpha=0.8, linewidth=0, antialiased=True)

    #Ruta de Newton en 3D 
    xs, ys = np.array(hist['x']), np.array(hist['y'])
    zs = f(xs, ys)
    ax.plot(xs, ys, zs, linewidth=2)
    ax.scatter(xs[-1], ys[-1], zs[-1], s=60)  #punto final en rojo

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')
    ax.set_title('Rosenbrock: superficie y ruta de Newton–Raphson')
    plt.tight_layout(); plt.savefig(filename, dpi=170); plt.close()

#Gráfica de contornos con ruta
def plot_contornos_con_ruta(hist, xlim=(-2, 2), ylim=(-1, 11),
                            nx=400, ny=400, filename='problema3a_contornos_ruta.png'):
    #Malla y niveles de contorno
    X = np.linspace(*xlim, nx)
    Y = np.linspace(*ylim, ny)
    XX, YY = np.meshgrid(X, Y)
    ZZ = f(XX, YY)

    plt.figure(figsize=(8, 6))
    cs = plt.contour(XX, YY, ZZ, levels=30)  #curvas de nivel de f
    plt.clabel(cs, inline=1, fontsize=8)

    #Ruta de Newton en el plano (x,y)
    xs, ys = np.array(hist['x']), np.array(hist['y'])
    plt.plot(xs, ys, marker='o', linewidth=2)
    plt.scatter(xs[-1], ys[-1], s=50)  #punto final

    plt.xlabel('x'); plt.ylabel('y')
    plt.title('Rosenbrock: contornos y ruta de Newton–Raphson')
    plt.tight_layout(); plt.savefig(filename, dpi=170); plt.close()

#Generar figuras
plot_superficie_con_ruta(hist)
plot_contornos_con_ruta(hist)
