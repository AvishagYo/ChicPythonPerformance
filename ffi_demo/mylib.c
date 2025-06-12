// mylib.c
#include <stdint.h>

__declspec(dllexport)
int64_t sum_range(int64_t n) {
    int64_t sum = 0;
    for (int64_t i = 0; i < n; ++i)
        sum += i;
    return sum;
}
