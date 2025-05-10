import numpy as np
import time
import ctypes
import os

print(f"NumPy version: {np.__version__}")

try:
    import numpy.distutils.system_info as sysinfo
    print("BLAS info:")
    print(sysinfo.get_info('blas_opt'))
except:
    print("Could not determine BLAS info.")

try:
    print("Linked OpenBLAS version:")
    print(np.__config__.show())
except:
    pass


print(f"Using NumPy version: {np.__version__}")

size = 3000
A = np.random.rand(size, size)
B = np.random.rand(size, size)

start = time.time()
C = np.dot(A, B)
end = time.time()

print(f"Execution time: {end - start:.4f} seconds")


print(f"Using NumPy version: {np.__version__}")

size = 1500
A = np.random.rand(size, size)

start = time.time()
U, S, Vt = np.linalg.svd(A, full_matrices=False)
end = time.time()

print(f"SVD time: {end - start:.4f} seconds")
