from __future__ import annotations
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#Definición simbólica
x = sp.Symbol('x', real=True)

#Función objetivo del problema
f_sym = 3*x**3 - 10*x**2 - 56*x + 50

#Derivadas analíticas
f1_sym = sp.diff(f_sym, x)   # f'(x)
f2_sym = sp.diff(f1_sym, x)  # f''(x)

f  = sp.lambdify(x, f_sym,  'numpy')
f1 = sp.lambdify(x, f1_sym, 'numpy')
f2 = sp.lambdify(x, f2_sym, 'numpy')

#Newton–Raphson para extremos
def newton_extremo_1d(f1, f2, x0: float, alpha: float = 1.0,
                      tol: float = 1e-8, max_iter: int = 100):
 
    xk = float(x0)            #Convertimos a float por seguridad
    history = [xk]            #Guardamos la trayectoria para graficar/analizar

    for k in range(max_iter):
        g = f1(xk)            #g = f'(x_k)    (debe tender a 0 en el óptimo)
        H = f2(xk)            #H = f''(x_k)   (curvatura local)

        #Si H ~ 0 o no es finito, no podemos dar un paso de Newton
        if not np.isfinite(H) or abs(H) < 1e-14:
            return xk, k, False, history

        #Paso de Newton con amortiguación
        step = - alpha * g / H
        xk = xk + step
        history.append(float(xk))

        #Criterio de parada: gradiente pequeño
        if abs(f1(xk)) < tol:
            return xk, k + 1, True, history

        #Evitamos desbordes numéricos
        if not np.isfinite(xk) or abs(xk) > 1e6:
            return xk, k + 1, False, history

    #no se logró convergencia en max_iter
    return xk, max_iter, False, history

def clasificar_extremo(f2, xc: float, eps: float = 1e-8) -> str:
    """Clasifica el punto crítico con el signo de f''(xc)."""
    H = f2(xc)
    if H > eps:
        return "min"
    elif H < -eps:
        return "max"
    else:
        return "plano/indeterminado"

#Barrido de semillas en [-6, 6]
def explorar_intervalo(a: float = -6, b: float = 6, n_seeds: int = 31,
                       alpha: float = 0.8, tol: float = 1e-10, max_iter: int = 100):
    """
    Lanza Newton desde múltiples semillas equiespaciadas en [a,b].
    Agrupa soluciones por cercanía para reportar puntos críticos únicos.
    """
    seeds = np.linspace(a, b, n_seeds)
    hallados = []      #(xc, it, ok)
    trayectorias = []  #lista de historiales x_k

    for s in seeds:
        xc, it, ok, hist = newton_extremo_1d(f1, f2, s, alpha=alpha, tol=tol, max_iter=max_iter)
        hallados.append((xc, it, ok))
        trayectorias.append(hist)

    #Agrupar raíces únicas por tolerancia
    uniques = []
    for xc, it, ok in hallados:
        if not ok:
            continue
        if not any(abs(xc - u) <= 1e-6 for u in uniques):
            uniques.append(xc)
    uniques = sorted(uniques)

    resultados = []
    for xc in uniques:
        resultados.append({
            "x": float(xc),
            "f(x)": float(f(xc)),
            "tipo": clasificar_extremo(f2, xc)
        })
    return resultados, seeds, trayectorias

def plot_funcion_y_extremos(resultados, a=-6, b=6,
                            filename='problema1_funcion_extremos.png'):
    """Grafica f(x) y marca los puntos críticos encontrados."""
    X = np.linspace(a, b, 800)
    Y = f(X)

    plt.figure(figsize=(8, 5))
    plt.plot(X, Y, label='f(x)')
    #Dibujar puntos críticos
    for r in resultados:
        marker = {'min': 'o', 'max': 's'}.get(r['tipo'], 'x')
        plt.plot(r['x'], r['f(x)'], marker=marker, markersize=8, label=r['tipo'])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Problema 1: f(x) y puntos críticos (Newton–Raphson)')
    #Evitar duplicados en la leyenda
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()

def plot_trayectorias(trayectorias, filename='problema1_trayectorias.png'):
    """Grafica la evolución de x_k para cada semilla."""
    plt.figure(figsize=(8, 5))
    for hist in trayectorias:
        plt.plot(range(len(hist)), hist, alpha=0.6)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Iteración k')
    plt.ylabel('x_k')
    plt.title('Trayectorias de Newton–Raphson desde diferentes semillas')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()

# ================================== 5) main ==================================
def main():
    
    a, b      = -6, 6
    n_seeds   = 31
    alpha     = 0.8     #amortiguación
    tol       = 1e-10
    max_iter  = 100

    resultados, seeds, trayectorias = explorar_intervalo(
        a=a, b=b, n_seeds=n_seeds, alpha=alpha, tol=tol, max_iter=max_iter
    )

    #Reporte en consola
    print("Puntos críticos encontrados (únicos):")
    for r in resultados:
        print(f"x* = {r['x']:.10f}\t f(x*) = {r['f(x)']:.10f}\t tipo = {r['tipo']}")

    #Guardar figuras
    plot_funcion_y_extremos(resultados, a=a, b=b,
                            filename='problema1_funcion_extremos.png')
    plot_trayectorias(trayectorias, filename='problema1_trayectorias.png')

    if resultados:
        mins = [r for r in resultados if r['tipo'] == 'min']
        maxs = [r for r in resultados if r['tipo'] == 'max']
        print("\nResumen:")
        print(f"- Total de puntos críticos: {len(resultados)}.  "
              f"Mínimos: {len(mins)}, Máximos: {len(maxs)}.")
        if mins:
            m = min(mins, key=lambda r: r['f(x)'])
            print(f"- Mínimo con menor f(x):  x ≈ {m['x']:.6f}, f(x) ≈ {m['f(x)']:.6f}")
        if maxs:
            M = max(maxs, key=lambda r: r['f(x)'])
            print(f"- Máximo con mayor f(x):  x ≈ {M['x']:.6f}, f(x) ≈ {M['f(x)']:.6f}")
    else:
        print("No hubo convergencia con los parámetros actuales. "
              "Prueba variando alpha, tol, semillas o el intervalo.")

if __name__ == "__main__":
    main()
