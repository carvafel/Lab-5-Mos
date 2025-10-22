import numpy as np
import matplotlib.pyplot as plt
import os

# Crear carpeta para guardar resultados
output_dir = "problema5_resultados"
os.makedirs(output_dir, exist_ok=True)


# 1. Generar los datos

np.random.seed(0)
N = 400
X = np.linspace(-2*np.pi, 2*np.pi, N).reshape(1, -1)
Y = np.sin(X)

# División entrenamiento / validación
perm = np.random.permutation(N)
train_idx = perm[:320]
val_idx = perm[320:]
X_train, Y_train = X[:, train_idx], Y[:, train_idx]
X_val, Y_val = X[:, val_idx], Y[:, val_idx]


# 2. Red neuronal simple (1 capa)
class SimpleNN:
    def __init__(self, n_in, n_hidden, n_out):
        self.W1 = np.random.randn(n_hidden, n_in) * np.sqrt(2/n_in)
        self.b1 = np.zeros((n_hidden,1))
        self.W2 = np.random.randn(n_out, n_hidden) * np.sqrt(2/n_hidden)
        self.b2 = np.zeros((n_out,1))

    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        a1 = np.tanh(z1)
        z2 = self.W2 @ a1 + self.b2
        return z1, a1, z2

    def predict(self, x):
        return self.forward(x)[2]

    def mse(self, preds, y):
        return np.mean((preds - y)**2)

    def gradients(self, x, y):
        z1, a1, z2 = self.forward(x)
        m = x.shape[1]
        dz2 = (z2 - y)
        dW2 = (dz2 @ a1.T) / m
        db2 = np.mean(dz2, axis=1, keepdims=True)
        da1 = self.W2.T @ dz2
        dz1 = da1 * (1 - np.tanh(z1)**2)
        dW1 = (dz1 @ x.T) / m
        db1 = np.mean(dz1, axis=1, keepdims=True)
        return [dW1, db1, dW2, db2]

# 3. Entrenamiento 
def train_gd(model, X, Y, epochs=500, lr=0.01, batch_size=32, X_val=None, Y_val=None):
    n = X.shape[1]
    losses = []
    val_losses = []
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X[:, idx]
            y_batch = Y[:, idx]
            dW1, db1, dW2, db2 = model.gradients(x_batch, y_batch)
            model.W1 -= lr * dW1
            model.b1 -= lr * db1
            model.W2 -= lr * dW2
            model.b2 -= lr * db2
        losses.append(model.mse(model.predict(X), Y))
        if X_val is not None:
            val_losses.append(model.mse(model.predict(X_val), Y_val))
    return losses, val_losses


# 4. Entrenamiento (descenso con momento)
def train_momentum(model, X, Y, epochs=500, lr=0.01, mu=0.9, batch_size=32, X_val=None, Y_val=None):
    n = X.shape[1]
    losses = []
    val_losses = []
    vW1 = np.zeros_like(model.W1)
    vb1 = np.zeros_like(model.b1)
    vW2 = np.zeros_like(model.W2)
    vb2 = np.zeros_like(model.b2)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X[:, idx]
            y_batch = Y[:, idx]
            dW1, db1, dW2, db2 = model.gradients(x_batch, y_batch)
            vW1 = mu * vW1 - lr * dW1
            vb1 = mu * vb1 - lr * db1
            vW2 = mu * vW2 - lr * dW2
            vb2 = mu * vb2 - lr * db2
            model.W1 += vW1
            model.b1 += vb1
            model.W2 += vW2
            model.b2 += vb2
        losses.append(model.mse(model.predict(X), Y))
        if X_val is not None:
            val_losses.append(model.mse(model.predict(X_val), Y_val))
    return losses, val_losses

# 5. Entrenar y comparar
hidden = 30
epochs = 500
lr = 0.02
batch = 32

model_gd = SimpleNN(1, hidden, 1)
loss_gd, val_gd = train_gd(model_gd, X_train, Y_train, epochs=epochs, lr=lr, batch_size=batch, X_val=X_val, Y_val=Y_val)

model_m = SimpleNN(1, hidden, 1)
loss_mom, val_mom = train_momentum(model_m, X_train, Y_train, epochs=epochs, lr=lr, mu=0.9, batch_size=batch, X_val=X_val, Y_val=Y_val)


# 6.Resultados
plt.figure(figsize=(8,5))
plt.plot(loss_gd, label='GD train loss')
plt.plot(loss_mom, label='Momentum train loss')
plt.yscale('log')
plt.xlabel('Época')
plt.ylabel('MSE (log)')
plt.title('Pérdida de entrenamiento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "perdida_entrenamiento.png"))
plt.close()

# Predicciones sobre todo el rango
xs = np.linspace(-2*np.pi, 2*np.pi, 400).reshape(1,-1)
true = np.sin(xs)
pred_gd = model_gd.predict(xs)
pred_m = model_m.predict(xs)

mse_gd = model_gd.mse(pred_gd, true)
mse_m = model_m.mse(pred_m, true)

plt.figure(figsize=(8,5))
plt.plot(xs.ravel(), true.ravel(), label='sin(x)', linewidth=2)
plt.plot(xs.ravel(), pred_gd.ravel(), label=f'GD (MSE={mse_gd:.4f})')
plt.plot(xs.ravel(), pred_m.ravel(), label=f'Momentum (MSE={mse_m:.4f})')
plt.legend()
plt.title('Aproximación de sin(x)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "aproximacion_seno.png"))
plt.close()

print(f"MSE final GD: {mse_gd:.6f}")
print(f"MSE final con Momento: {mse_m:.6f}")
print(f"\nGráficas guardadas en la carpeta: '{output_dir}'")
