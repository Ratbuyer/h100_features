// This code uses TMA's 2d load to load a matrix's tile to
// shared memory and then change the value in the
// shared memory and uses TMA's store to store the
// tile back to global memory. We print the result matrix to prove the
// changes are done

#include <cuda/barrier>
#include <cuda/std/utility> // cuda::std::move
#include <stdio.h>
#include <cuda.h>

#include "test_macros.cuh"
#include "tma_tensor_map.cuh"
#include "matrix_utilities.cuh"
#include "tma.cuh"

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

constexpr size_t M = 64; // Width of tensor (in # elements)
constexpr size_t K = 32; // Height of tensor (in # elements)
constexpr size_t gmem_len = M * K;

constexpr int m = 16; // Width of shared memory buffer (in # elements)
constexpr int k = 8; // Height of shared memory buffer (in # elements)

static constexpr int buf_len = k * m;

__global__ void test(const __grid_constant__ CUtensorMap global_fake_tensor_map, int base_i, int base_j)
{
  __shared__ alignas(128) int smem_buffer[buf_len];
  __shared__ barrier bar;

  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  // Load data:
  uint64_t token;
  if (threadIdx.x == 0)
  {
    // just to demonstrate prefetch
    // copy_async_2d_prefetch(global_fake_tensor_map, base_j, base_i);
    // call the loading api
    cde::cp_async_bulk_tensor_2d_global_to_shared(smem_buffer, &global_fake_tensor_map, base_j, base_i, bar);
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  }
  else
  {
    token = bar.arrive();
  }

  bar.wait(cuda::std::move(token));

  __syncthreads();

  // Update smem, change from 1 to 2
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x)
  {
    smem_buffer[i] += 1;
  }

  cde::fence_proxy_async_shared_cta();
  __syncthreads();

  // Write back to global memory:
  if (threadIdx.x == 0)
  {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&global_fake_tensor_map, base_j, base_i, smem_buffer);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }
  __threadfence();
  __syncthreads();
}

int main()
{
  // set host matrix to 1
  int host_tensor[gmem_len];
  for (int i = 0; i < gmem_len; i++)
  {
    host_tensor[i] = 1;
  }

  // copy host matrix to device
  int *tensor_ptr = nullptr;
  cudaMalloc(&tensor_ptr, gmem_len * sizeof(int));
  cudaMemcpy(tensor_ptr, host_tensor, gmem_len * sizeof(int), cudaMemcpyHostToDevice);

  // create tensor map for the matrix
  CUtensorMap tensor_map = create_2d_tensor_map(M, K, m, k, tensor_ptr);

  // launch kernel, select a tile coordinate
  int tile_i = 16;
  int tile_j = 0;
  test<<<1, 128>>>(tensor_map, tile_i, tile_j);

  cudaDeviceSynchronize();

  // check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

  // copy device matrix to host
  int host_gmem_tensor[gmem_len];
  cudaMemcpy(host_gmem_tensor, tensor_ptr, gmem_len * sizeof(int), cudaMemcpyDeviceToHost);

  // verify the results
  print_matrix(host_gmem_tensor, M, K);

  return 0;
}