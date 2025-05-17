from numba import njit
import numpy as np
from skimage import io
from time import time

IMAGE = io.imread("hallway.jpg")

# Code that is decorated with numba.njit looks like Python code,
# but is actually compiled into machine code, at runtime. This
# is fast, low-level code!
@njit
def dither(img):
    # Allow negative values and wider range than a uint8 has:
    result = img.astype(np.int16)
    y_size = img.shape[0]
    x_size = img.shape[1]
    last_y = y_size - 1
    last_x = x_size - 1
    for y in range(y_size):
        for x in range(x_size):
            old_value = result[y, x]
            if old_value < 0:
                new_value = 0
            elif old_value > 255:
                new_value = 255
            else:
                new_value = np.uint8(np.round(old_value / 255.0)) * 255
            result[y, x] = new_value
            # We might get a negative value for the error:
            error = np.int16(old_value) - new_value
            if x < last_x:
                result[y, x + 1] += error * 7 // 16
            if y < last_y and x > 0:
                result[y + 1, x - 1] += error * 3 // 16
            if y < last_y:
                result[y + 1, x] += error * 5 // 16
            if y < last_y and x < last_x:
                result[y + 1, x + 1] += error // 16

    return result.astype(np.uint8)

# Run a first time, to compile the code:
dither(IMAGE)

# Make sure we run this for long enough that the profiler
# gets sufficient samples:
start = time()
runs = 0
while time() - start < 5:
    runs += 1
    dither(IMAGE)

elapsed = time() - start
print(f"Processed {int(round(runs / elapsed))} images / sec")