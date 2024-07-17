// This code combines the features on H100 to do gemm.

#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <random>
#include <iostream>
#include <cassert>

#include "../headers/device/descriptor.cuh"
#include "../headers/host/matrix_utilities.cuh"
#include "../headers/device/wgmma.cuh"

const int M = 512;
const int N = 512;
const int K = 512;

const int M2 = 64;
const int N2 = 8;
const int K2 = 16;

__global__ void gemm(half *A, half *B, half *C)
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
      A_shared[offset] = A[(block_id_m * M2 + i) * K + k_step * K2 + j];
    }

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
  half h_C[M * N];
  half h_CPU[M * N];
  half h_A[M * K];
  // half h_A_reordered[M * K];
  half h_B[K * N];

  fill_random(h_A, M, K);
  fill_random(h_B, K, N);

  half *d_A, *d_B;

  cudaMalloc((void **)&d_A, M * K * sizeof(half));
  cudaMalloc((void **)&d_B, K * N * sizeof(half));
  cudaMalloc((void **)&d_C, M * N * sizeof(half));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

  const int threads_per_block = 32 * 4; // 4 warps
  const int blocks = (M / M2) * (N / N2);

  gemm<<<blocks, threads_per_block>>>(d_A, d_B, d_C);

  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

  // print_matrix(h_C, M, N);

  CPU_gemm(h_A, h_B, h_CPU, M, N, K);

  compare_matrices(h_C, h_CPU, M, N);

  return 0;
}