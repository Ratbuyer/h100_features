/*
This code uses TMA's 1d tensor load to load
a portion of an array to shared memory and then
change the value in the shared memory and uses TMA's store
to store the portion back to global memory. We print the result
to show the changes are done.
*/

// supress warning about barrier in shared memory on line 32
#pragma nv_diag_suppress static_var_with_dynamic_init

#include <cuda/barrier>
#include <iostream>

#include "tma_tensor_map.cuh"
#include "matrix_utilities.cuh"
#include "profile_utilities.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

const int array_size = 128;
const int tile_size = 16;

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int coordinate)
{
  // Shared memory buffers for tile. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int tile_shared[tile_size];

  // 4. change the value in shared memory
  for (int i = threadIdx.x; i < array_size; i += blockDim.x)
  {
    if (i < tile_size)
    {
      tile_shared[i] = 2;
    }
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0)
  {
    asm volatile(
        "cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group "
        "[%0, {%1}], [%2];\n"
        :
        : "l"(tensor_map),
          "r"(coordinate),
          "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(tile_shared)))
        : "memory");
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cde::cp_async_bulk_wait_group_read<0>();
  }

  __threadfence();
  __syncthreads();
}

int main()
{
  // initialize array and fill it with values
  int h_data[array_size];
  for (size_t i = 0; i < array_size; ++i)
  {
    h_data[i] = 1;
  }

  // print the array before the kernel
  // one tile per line
  print_matrix(h_data, array_size / tile_size, tile_size);

  // transfer array to device
  int *d_data = nullptr;
  cudaMalloc(&d_data, array_size * sizeof(int));
  cudaMemcpy(d_data, h_data, array_size * sizeof(int), cudaMemcpyHostToDevice);

  // create tensor map
  CUtensorMap tensor_map = create_1d_tensor_map(array_size, tile_size, d_data);

  size_t offset = tile_size * 3; // select the second tile of the array to change
  kernel<<<1, 128>>>(tensor_map, offset);

  cuda_check_error();

  cudaMemcpy(h_data, d_data, array_size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_data);

  // print the array after the kernel
  print_matrix(h_data, array_size / tile_size, tile_size);

  return 0;
}