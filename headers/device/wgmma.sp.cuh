#include <stdint.h>

__device__ void wgmma_sp_async(
    uint32_t *c, const uint64_t desc_a, const uint64_t desc_b, const uint32_t metadata
)
{
  asm volatile("wgmma.mma_async.sp.sync.aligned.m64n8k32.f16.f16.f16 "
               "{%0, %1}, " // c
               "%2, %3, "   // desc A, B
               "%4, "       // meta
               "0, "       // thread selection
               "1, "       // scale D
               "%7, %8, "   // +/- scale A, B
               "%9, %10;"   // transpose A, B
               : "+r"(c[0]), "+r"(c[1])
               : "l"(desc_a), "l"(desc_b),
                 "r"(metadata),   // metadata
                 "r"(0),        // thread selection
                 "r"(1),          // scale D
                 "n"(1), "n"(1),  // +- scale A, B
                 "n"(0), "n"(1)); // transpose A, B
}