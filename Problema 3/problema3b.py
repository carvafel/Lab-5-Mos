import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import csv

#Definición simbólica
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4', real=True)
x_vec = sp.Matrix([x1, x2, x3, x4])
c = sp.Matrix([1, 2, 3, 4])

#construir f con suma de Python
f_sym = sum((x_vec[j] - c[j])**2 for j in range(4))

grad_sym = sp.Matrix([sp.diff(f_sym, v) for v in x_vec])   #∇f(x)
hess_sym = sp.hessian(f_sym, (x1, x2, x3, x4))             #∇²f(x)

#Funciones numéricas
f    = sp.lambdify((x1, x2, x3, x4), f_sym,    'numpy')
grad = sp.lambdify((x1, x2, x3, x4), grad_sym, 'numpy')
hess = sp.lambdify((x1, x2, x3, x4), hess_sym, 'numpy')

#Newton–Raphson 4D
def newton_4d(f, grad, hess, x0, alpha=1.0, tol=1e-10, max_iter=100):
    xk = np.asarray(x0, dtype=float).reshape(4, 1)
    X_hist, f_hist, g_hist = [xk.copy()], [float(f(*xk.flatten()))], []

    for k in range(max_iter):
        gk = np.array(grad(*xk.flatten()), dtype=float).reshape(4, 1)
        Hk = np.array(hess(*xk.flatten()), dtype=float)
        gn = float(np.linalg.norm(gk, 2))
        g_hist.append(gn)
        if gn < tol:
            break
        try:
            pk = np.linalg.solve(Hk, gk)  #Hk p = gk
        except np.linalg.LinAlgError:
            print(f"[Aviso] Hessiana singular/mal condicionada en k={k}.")
            break
        xk = xk - alpha * pk
        X_hist.append(xk.copy())
        f_hist.append(float(f(*xk.flatten())))

    return {"X": X_hist, "f": f_hist, "grad_norm": g_hist, "iters": len(X_hist)-1}

#Ejecutar
x0 = np.array([5.0, -5.0, 0.0, 10.0])  #prueba
alpha = 1.0  #con Hessiana 2I, converge en 1 paso
hist = newton_4d(f, grad, hess, x0=x0, alpha=alpha, tol=1e-10, max_iter=20)

print("Iteraciones realizadas:", hist["iters"])
xf = hist["X"][-1].flatten()
print("x_final:", xf)
print("f(x_final):", hist["f"][-1])
print("||grad|| final:",
      hist["grad_norm"][-1] if hist["grad_norm"] else np.linalg.norm(grad(*xf), 2))

#Guardar CSV
with open("problema3b_iteraciones.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp)
    w.writerow(["k", "x1", "x2", "x3", "x4", "f(x)", "||grad||_2"])
    for k in range(len(hist["X"])):
        xk = hist["X"][k].flatten()
        gk = np.array(grad(*xk), dtype=float).reshape(4,1)
        w.writerow([k, *xk, hist["f"][k], float(np.linalg.norm(gk, 2))])

#Gráficas
#Componentes vs. k
plt.figure(figsize=(8, 5))
Xs = np.array([x.flatten() for x in hist["X"]])
for j, label in enumerate(["x1", "x2", "x3", "x4"]):
    plt.plot(range(len(Xs)), Xs[:, j], marker="o", linewidth=2, label=label)
plt.xlabel("Iteración k"); plt.ylabel("Componente")
plt.title("Problema 3(b): componentes de x_k vs. iteración")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("problema3b_componentes_vs_k.png", dpi=160)
plt.close()

#Norma del gradiente (log)
plt.figure(figsize=(8, 5))
g = np.array(hist["grad_norm"], dtype=float)
if g.size > 0:
    plt.semilogy(range(len(g)), g, marker="o", linewidth=2)
    plt.ylabel("||∇f(x_k)||_2 (log)")
else:
    plt.semilogy([0], [np.nan], marker="o")
    plt.ylabel("||∇f(x_k)||_2")
plt.xlabel("Iteración k")
plt.title("Problema 3(b): norma del gradiente")
plt.grid(True, which="both"); plt.tight_layout()
plt.savefig("problema3b_normagrad_vs_k.png", dpi=160)
plt.close()
