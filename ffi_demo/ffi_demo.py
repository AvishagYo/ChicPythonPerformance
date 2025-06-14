import ctypes
import time
import os
import platform

# Load DLL
if platform.system() == "Windows":
    os.add_dll_directory(os.getcwd())
    lib = ctypes.CDLL("mylib.dll")
    badlib = ctypes.CDLL("mylibbad.dll")
else:
    exit(1)

# Set up the C function
lib.sum_range.argtypes = [ctypes.c_int64]
lib.sum_range.restype = ctypes.c_int64

# Set up the C function
badlib.add.argtypes = [ctypes.c_int64, ctypes.c_int64]
badlib.add.restype = ctypes.c_int64


# Python version
def python_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total


# Python version
def bad_ffi_sum(n):
    total = 0
    for i in range(n):
        total = badlib.add(total, i)
    return total

# Compare performance
n = 10_000_000

start = time.time()
res_py = python_sum(n)
py_time = time.time() - start


start = time.time()
res_bad_c = bad_ffi_sum(n)
bad_c_time = time.time() - start


start = time.time()
res_c = lib.sum_range(n)
c_time = time.time() - start

print(f"Python: {py_time:.6f} s, result = {res_py}")
print(f"Bad C FFI : {bad_c_time:.6f} s, result = {res_bad_c}")
print(f"C FFI : {c_time:.6f} s, result = {res_c}")
print(f"Speedup: {py_time / c_time:.2f}x")
