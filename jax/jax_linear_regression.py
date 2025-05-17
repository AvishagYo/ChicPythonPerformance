import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data
key = jax.random.PRNGKey(0)
x = jnp.linspace(-5, 5, 100)
true_w, true_b = 2.0, -1.0
y = true_w * x + true_b + jax.random.normal(key, (100,)) * 1.0

# Model: y = w * x + b
def model(params, x):
    w, b = params
    return w * x + b

# Mean squared error loss
def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

# Gradient of loss function
grad_loss = grad(loss_fn)

# JIT-compiled update step
@jit
def update(params, x, y, lr=0.01):
    grads = grad_loss(params, x, y)
    return [p - lr * g for p, g in zip(params, grads)]

# Initialize parameters
params = [jnp.array(0.0), jnp.array(0.0)]  # w, b

# Training loop
for i in range(300):
    params = update(params, x, y)
    if i % 50 == 0:
        current_loss = loss_fn(params, x, y)
        print(f"Step {i}: loss = {current_loss:.4f}, w = {params[0]:.4f}, b = {params[1]:.4f}")

# Plot results
preds = model(params, x)
plt.scatter(x, y, label="Data")
plt.plot(x, preds, color="red", label="Fitted Line")
plt.legend()
plt.title("JAX Linear Regression Demo")
plt.show()
