import numpy as np
import jax
import jax.numpy as jnp
import time

# Matrix size and number of iterations
size = 1000
iterations = 10

print(f"\nMatrix multiplication benchmark (size={size}, iterations={iterations})")

# Prepare data
np_A = np.random.rand(size, size)
np_B = np.random.rand(size, size)

jax_A = jnp.array(np_A)
jax_B = jnp.array(np_B)

### NumPy benchmark
start = time.perf_counter()
for _ in range(iterations):
    np.dot(np_A, np_B)
end = time.perf_counter()
print(f"NumPy: {end - start:.4f} seconds")

### JAX without JIT
start = time.perf_counter()
for _ in range(iterations):
    jnp.dot(jax_A, jax_B).block_until_ready()  # ensure it's evaluated
end = time.perf_counter()
print(f"JAX (no JIT): {end - start:.4f} seconds")

### JAX with JIT
@jax.jit
def fast_dot(A, B):
    return jnp.dot(A, B)

# Warm-up (JIT compile)
fast_dot(jax_A, jax_B).block_until_ready()

start = time.perf_counter()
for _ in range(iterations):
    fast_dot(jax_A, jax_B).block_until_ready()
end = time.perf_counter()
print(f"JAX (with JIT): {end - start:.4f} seconds")
