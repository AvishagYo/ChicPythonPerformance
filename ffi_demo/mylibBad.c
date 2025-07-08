#include <stdint.h>
#ifdef _WIN32
__declspec(dllexport)
#endif
int64_t add(int64_t a, int64_t b) {
    return a + b;
}
