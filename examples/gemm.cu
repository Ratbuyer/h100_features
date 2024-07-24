// This code combines the features on H100 to do gemm.

#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <random>
#include <iostream>
#include <cassert>
#include <cuda/barrier>
#include <cudaTypedefs.h>

#include "descriptor.cuh"
#include "matrix_utilities.cuh"
#include "wgmma.cuh"
#include "tma_tensor_map.cuh"
#include "test_macros.cuh"
#include "kernel.cuh"

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

const int M = 512;
const int N = 512;
const int K = 512;

const int M2 = 64;
const int N2 = 8;
const int K2 = 16;

__global__ void gemm(half *A, half *B, half *C,
                     const __grid_constant__ CUtensorMap A_tensor_map)
{
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int group_id = lane_id >> 2;
  const int lane_in_group = lane_id & 3;

  const int block_id = blockIdx.x;
  const int block_id_m = block_id / (N / N2);
  const int block_id_n = block_id % (N / N2);

  uint32_t c[2] = {0};

  __align__(128) __shared__ half A_buffer[M2 * K2];
  // __align__(16) __shared__ half B_buffer[K2 * N2];

  __align__(16) __shared__ half A_shared[M2 * K2];
  __align__(16) __shared__ half B_shared[K2 * N2];

  GmmaDescriptor desc_a = make_desc_a(A_shared);
  GmmaDescriptor desc_b = make_desc_b(B_shared);

  for (int k_step = 0; k_step < K / K2; k_step++)
  {

    // if (tid == 0)
    // {
    //   for (int i = 0; i < M2; i++)
    //   {
    //     for (int j = 0; j < K2; j++)
    //     {
    //       int block_x = i / 8;
    //       int block_row = i % 8;
    //       int block_y = j / 8;
    //       int block_col = j % 8;
    //       int block_id = block_x * 2 + block_y;
    //       int offset = block_id * 64 + block_row * 8 + block_col;
    //       A_shared[offset] = A[(block_id_m * M2 + i) * K + k_step * K2 + j];
    //     }
    //   }

    //   for (int i = 0; i < K2; i++)
    //   {
    //     for (int j = 0; j < N2; j++)
    //     {
    //       int block_x = i / 8;
    //       int block_row = i % 8;
    //       int block_y = j / 8;
    //       int block_col = j % 8;
    //       int block_id = block_x * 1 + block_y;
    //       int offset = block_id * 64 + block_row * 8 + block_col;
    //       B_shared[offset] = B[(k_step * K2 + i) * N + block_id_n * N2 + j];
    //     }
    //   }
    // }

    __shared__ barrier bar;

    if (threadIdx.x == 0)
    {
      init(&bar, blockDim.x);
    }
    __threadfence();
    __syncthreads();

    // load a using tma
    uint64_t token;
    if (threadIdx.x == 0)
    {
      // Fastest moving coordinate first.
      cde::cp_async_bulk_tensor_2d_global_to_shared(A_buffer, &A_tensor_map, k_step * K2, block_id_m * M2, bar);
      token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(A_buffer));
    }
    else
    {
      token = bar.arrive();
    }

    bar.wait(cuda::std::move(token));

    __threadfence();
    __syncthreads();

    for (int index = tid * 8; index < (tid + 1) * 8 && index < 64 * 16; index++)
    {
      int i = index / 16;
      int j = index % 16;

      int block_x = i / 8;
      int block_row = i % 8;
      int block_y = j / 8;
      int block_col = j % 8;
      int block_id = block_x * 2 + block_y;
      int offset = block_id * 64 + block_row * 8 + block_col;
      A_shared[offset] = A_buffer[i * K2 + j];
    }

    cde::fence_proxy_async_shared_cta();

    __threadfence();
    __syncthreads();

    {
      int i = tid / 8;
      int j = tid % 8;

      int block_x = i / 8;
      int block_row = i % 8;
      int block_y = j / 8;
      int block_col = j % 8;
      int block_id = block_x * 1 + block_y;
      int offset = block_id * 64 + block_row * 8 + block_col;
      B_shared[offset] = B[(k_step * K2 + i) * N + block_id_n * N2 + j];
    }

    __threadfence();
    __syncthreads();

    warpgroup_fence_operand(c[0]);
    warpgroup_fence_operand(c[1]);

    warpgroup_arrive();

    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
                 "{%0, %1}, " // c
                 "%2, %3, "   // a, b
                 "1, "        // scale_c
                 "1, 1, "     // + or - a, b
                 "0, 1;"      // trans a, trans b
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b));

    warpgroup_commit_batch();

    warpgroup_wait<0>();

    warpgroup_fence_operand(c[0]);
    warpgroup_fence_operand(c[1]);

    __threadfence();
    __syncthreads();
  }

  // store back to c
  uint32_t *C_ptr = reinterpret_cast<uint32_t *>(C);

  int offset1 = block_id_m * M2 * (N / 2) + warp_id * 16 * (N / 2) + group_id * (N / 2) + block_id_n * (N2 / 2) + lane_in_group;
  int offset2 = block_id_m * M2 * (N / 2) + warp_id * 16 * (N / 2) + (group_id + 8) * (N / 2) + block_id_n * (N2 / 2) + lane_in_group;

  warpgroup_arrive();

  C_ptr[offset1] = c[0];
  C_ptr[offset2] = c[1];
}

int main()
{

  assert(M % M2 == 0);
  assert(N % N2 == 0);
  assert(K % K2 == 0);

  half *d_C;
  half h_C[M * N]{};
  half h_CPU[M * N]{};
  half h_A[M * K];
  half h_B[K * N];

  // fill_fixed(h_A, M, K, 1);
  // fill_fixed(h_B, K, N, 1);

  fill_random(h_A, M, K);
  fill_random(h_B, K, N);

  half *d_A = nullptr;
  half *d_B = nullptr;

  cudaMalloc((void **)&d_A, M * K * sizeof(half));
  cudaMalloc((void **)&d_B, K * N * sizeof(half));
  cudaMalloc((void **)&d_C, M * N * sizeof(half));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

  // initalize tensor map for tma
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  CUtensorMap A_tensor_descriptor{};

  // both A and B are matrices (2d)
  const int rank = 2;

  uint64_t size[rank] = {M, K};
  uint64_t stride[rank - 1] = {K * sizeof(half)};
  uint32_t box_size[rank] = {K2, M2};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult res = cuTensorMapEncodeTiled(
      &A_tensor_descriptor, // CUtensorMap *tensorMap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      rank,        // cuuint32_t tensorRank,
      d_A,         // void *globalAddress,
      size,        // const cuuint64_t *globalDim,
      stride,      // const cuuint64_t *globalStrides,
      box_size,    // const cuuint32_t *boxDim,
      elem_stride, // const cuuint32_t *elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  assert(res == CUDA_SUCCESS && "tensormap creation failed.");

  const int threads_per_block = 32 * 4; // 4 warps
  const int blocks = (M / M2) * (N / N2);

  gemm<<<blocks, threads_per_block>>>(d_A, d_B, d_C, A_tensor_descriptor);

  cuda_check_error();

  cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

  // print_matrix(h_C, M, N);

  CPU_gemm(h_A, h_B, h_CPU, M, N, K);

  compare_matrices(h_CPU, h_C, M, N);

  return 0;
}