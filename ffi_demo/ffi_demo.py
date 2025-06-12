import ctypes
import time
import os
import platform

# Load DLL
if platform.system() == "Windows":
    os.add_dll_directory(os.getcwd())
    lib = ctypes.CDLL("mylib.dll")
else:
    lib = ctypes.CDLL("./mylib.so")

# Set up the C function
lib.sum_range.argtypes = [ctypes.c_int64]
lib.sum_range.restype = ctypes.c_int64

# Python version
def python_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total

# Compare performance
n = 10_000_000

start = time.time()
res_py = python_sum(n)
py_time = time.time() - start

start = time.time()
res_c = lib.sum_range(n)
c_time = time.time() - start

print(f"Python: {py_time:.6f} s, result = {res_py}")
print(f"C FFI : {c_time:.6f} s, result = {res_c}")
print(f"Speedup: {py_time / c_time:.2f}x")
