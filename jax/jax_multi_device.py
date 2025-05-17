import jax
import jax.numpy as jnp
import time

# List devices
devices = jax.devices()
print(f"\nFound {len(devices)} device(s):")
for i, d in enumerate(devices):
    print(f"  {i}: {d}")

# Simple computation to parallelize
def heavy_compute(x):
    return jnp.sin(x) * jnp.exp(x)

# Parallel version using pmap
@jax.pmap
def parallel_heavy_compute(x):
    return heavy_compute(x)

# Prepare input data — one chunk per device
num_devices = len(devices)
x = jnp.linspace(0, 10, 10_000)
x_split = jnp.reshape(x, (num_devices, -1))  # shape: [num_devices, batch_per_device]

# Time it
start = time.perf_counter()
result = parallel_heavy_compute(x_split)
jax.block_until_ready(result)
end = time.perf_counter()

print(f"\nParallel computation across {num_devices} devices took: {end - start:.4f} seconds")
