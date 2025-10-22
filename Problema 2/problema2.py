import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#Variable simbólica
x = sp.Symbol('x', real=True)

#Función objetivo del problema
f_sym = x**5 - 8*x**3 + 10*x + 6

#Derivadas analíticas
f1_sym = sp.diff(f_sym, x)   #Primera derivada f'(x)
f2_sym = sp.diff(f1_sym, x)  #Segunda derivada f''(x)

#Conversión a funciones numéricas 
f  = sp.lambdify(x, f_sym,  'numpy')
f1 = sp.lambdify(x, f1_sym, 'numpy')
f2 = sp.lambdify(x, f2_sym, 'numpy')


#Método de Newton–Raphson para encontrar extremos locales

def newton_extremo_1d(f1, f2, x0, alpha=0.8, tol=1e-10, max_iter=200):
    """
    Aplica el método de Newton–Raphson para resolver f'(x)=0, es decir,
    encontrar puntos críticos de f(x).
    """

    xk = float(x0)          #Inicialización
    history = [xk]

    for k in range(max_iter):
        g = f1(xk)          #Derivada primera f'(x_k)
        H = f2(xk)          #Derivada segunda f''(x_k)

        #Evitar división por cero o Hessiana mal condicionada
        if not np.isfinite(H) or abs(H) < 1e-14:
            return xk, k, False, history

        #Paso de Newton amortiguado
        step = - alpha * g / H
        xk = xk + step
        history.append(float(xk))

        #Criterio de convergencia: derivada próxima a cero
        if abs(f1(xk)) < tol:
            return xk, k+1, True, history

        #Prevención de valores no finitos o divergentes
        if not np.isfinite(xk) or abs(xk) > 1e6:
            return xk, k+1, False, history

    #Si no converge en max_iter, se retorna el último valor
    return xk, max_iter, False, history


def clasificar_extremo(f2, xc, eps=1e-8):
    """
    Clasifica el punto crítico según el signo de la segunda derivada.
    """
    H = f2(xc)
    if H > eps:
        return "min"   #Mínimo local
    elif H < -eps:
        return "max"   #Máximo local
    else:
        return "plano/indeterminado"


#Barrido de semillas en [-3, 3]

def explorar_intervalo(a=-3, b=3, n_seeds=49, alpha=0.8, tol=1e-10, max_iter=200):
    """
    Lanza el método desde múltiples valores iniciales para asegurar
    la detección de todos los puntos críticos en el intervalo [-3,3].
    """
    seeds = np.linspace(a, b, n_seeds)
    hallados = []
    trayectorias = []

    for s in seeds:
        xc, it, ok, hist = newton_extremo_1d(f1, f2, s, alpha, tol, max_iter)
        trayectorias.append(hist)
        if ok and (a - 1e-8) <= xc <= (b + 1e-8):
            hallados.append(xc)

    #Agrupar resultados únicos 
    unicos = []
    for xc in hallados:
        if not any(abs(xc - u) <= 1e-6 for u in unicos):
            unicos.append(xc)
    unicos.sort()

    resultados = []
    for xc in unicos:
        resultados.append({
            "x": float(xc),
            "f(x)": float(f(xc)),
            "tipo": clasificar_extremo(f2, xc)
        })

    return resultados, seeds, trayectorias


#Cálculo de extremos globales en [-3,3]

def extremos_globales(resultados, a=-3, b=3):
    """
    Evalúa f(x) en los bordes y en los puntos críticos encontrados,
    identificando el máximo y mínimo globales del intervalo.
    """
    candidatos = [(a, float(f(a)), "borde"), (b, float(f(b)), "borde")]
    for r in resultados:
        candidatos.append((r["x"], r["f(x)"], r["tipo"]))

    gmin = min(candidatos, key=lambda t: t[1])
    gmax = max(candidatos, key=lambda t: t[1])
    return gmin, gmax, candidatos


#Visualización de resultados

def plot_funcion_y_extremos(resultados, global_min, global_max,
                            a=-3, b=3, filename='problema2_funcion_extremos.png'):
    """
    Grafica la función f(x), los puntos críticos locales y
    marca los extremos globales dentro del intervalo.
    """
    X = np.linspace(a, b, 800)
    Y = f(X)

    plt.figure(figsize=(8, 5))
    plt.plot(X, Y, label='f(x)')

    #Puntos críticos locales
    for r in resultados:
        marker = {'min': 'o', 'max': 's'}.get(r['tipo'], 'x')
        plt.plot(r['x'], r['f(x)'], marker=marker, markersize=7, label=r['tipo'])

    #Puntos globales 
    plt.plot(global_min[0], global_min[1], marker='v', markersize=10, label='mínimo global')
    plt.plot(global_max[0], global_max[1], marker='^', markersize=10, label='máximo global')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Problema 2: f(x), extremos locales y globales')
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()


def plot_trayectorias(trayectorias, filename='problema2_trayectorias.png'):
    """
    Muestra cómo evoluciona x_k en cada iteración para las distintas semillas.
    """
    plt.figure(figsize=(8, 5))
    for hist in trayectorias:
        plt.plot(range(len(hist)), hist, alpha=0.55)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Iteración k')
    plt.ylabel('x_k')
    plt.title('Trayectorias de Newton–Raphson (Problema 2)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()



def main():
    a, b = -3, 3
    alpha = 0.8
    tol = 1e-10
    n_seeds = 49

    #Ejecución del método
    resultados, seeds, trayectorias = explorar_intervalo(a, b, n_seeds, alpha, tol)
    gmin, gmax, candidatos = extremos_globales(resultados, a, b)

    #Reporte de resultados
    print("Puntos críticos locales encontrados en [-3,3]:")
    for r in resultados:
        print(f"x* = {r['x']:.10f}\t f(x*) = {r['f(x)']:.10f}\t tipo = {r['tipo']}")

    print("\nEvaluación completa (incluyendo bordes):")
    for xk, fk, t in candidatos:
        print(f"x = {xk:.10f}\t f(x) = {fk:.10f}\t tipo = {t}")

    print("\nMínimo global:\n  x ≈ {:.10f},  f(x) ≈ {:.10f} ({})"
          .format(gmin[0], gmin[1], gmin[2]))
    print("Máximo global:\n  x ≈ {:.10f},  f(x) ≈ {:.10f} ({})"
          .format(gmax[0], gmax[1], gmax[2]))

    #Generar figuras
    plot_funcion_y_extremos(resultados, gmin, gmax, a, b)
    plot_trayectorias(trayectorias)



#main
if __name__ == "__main__":
    main()
