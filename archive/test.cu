#include <iostream>

enum Cache_Policy
{
  evict_normal,
};

__device__ void copy_async_prefetch(void * src, int size, Cache_Policy policy)
{
  asm volatile("cp.async.prefetch.global.L2::cache_hint [%0], %1;"
               :
               : "r"(src), "r"(size));
}

int main() {
    // Example usage
    void *src = nullptr; // Replace with actual source address
    int size = 1024; // Replace with actual size
    Cache_Policy policy = evict_normal;
    
    // Call the function (in the context of CUDA kernel)
    // copy_async_prefetch<<<1, 1>>>(src, size, policy);

    return 0;
}
