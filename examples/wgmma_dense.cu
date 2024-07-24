// This code demonstrates how to use the wgmma instructions

#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <random>
#include <iostream>

#include "matrix_utilities.cuh"
#include "profile_utilities.cuh"
#include "wgmma.cuh"

const int M = 64;
const int N = 8;
const int K = 16;

const int threads_per_block = 32 * 4; // 4 warps
const int blocks = 1;

__global__ void kernel(half *A, half *B, half *C)
{
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int group_id = lane_id >> 2;
  const int lane_in_group = lane_id & 3;

  __syncthreads();

  __align__(16) __shared__ half A_shared[M * K];
  __align__(16) __shared__ half B_shared[K * N];

  // 8x8 core blocks
  if (tid == 0)
  {

    for (int i = 0; i < M; i++)
    {
      for (int j = 0; j < K; j++)
      {
        int block_x = i / 8;
        int block_row = i % 8;
        int block_y = j / 8;
        int block_col = j % 8;
        int block_id = block_x * 2 + block_y;
        int offset = block_id * 64 + block_row * 8 + block_col;
        A_shared[offset] = A[i * K + j];
      }
    }

    for (int i = 0; i < K; i++)
    {
      for (int j = 0; j < N; j++)
      {
        int block_x = i / 8;
        int block_row = i % 8;
        int block_y = j / 8;
        int block_col = j % 8;
        int block_id = block_x * 1 + block_y;
        int offset = block_id * 64 + block_row * 8 + block_col;
        B_shared[offset] = B[i * N + j];
      }
    }
  }

  __syncthreads();

  GmmaDescriptor desc_a = make_desc_a(A_shared);
  GmmaDescriptor desc_b = make_desc_b(B_shared);

  uint32_t c[2] = {};

  asm volatile("wgmma.fence.sync.aligned; \n");

  // wgmma.mma_async.sync.aligned.shape.dtype.f16.f16  d, a-desc, b-desc, scale-d, imm-scale-a, imme-scale-b, imm-trans-a, imm-trans-b;
  // wgmma.mma_async.sync.aligned.shape.dtype.f16.f16  d, a, b-desc, scale-d, imm-scale-a, imme-scale-b, imm-trans-b;
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
               "{%0, %1}, "
               "%2, %3, "
               "1, "
               "1, 1, "
               "0, 1;"
               : "+r"(c[0]), "+r"(c[1])
               : "l"(desc_a), "l"(desc_b));

  asm volatile("wgmma.commit_group.sync.aligned; \n");

  asm volatile("wgmma.wait_group.sync.aligned 0; \n");

  __syncthreads();

  asm volatile("wgmma.fence.sync.aligned; \n");

  // half * reg_ptr = reinterpret_cast<half *>(&reg);

  // if (tid == 0) {
  //   printf("%f\n", __half2float(reg_ptr[0]));
  // }

  uint32_t *C_ptr = reinterpret_cast<uint32_t *>(C);

  int offset1 = warp_id * 16 * 4 + group_id * 4 + lane_in_group;
  int offset2 = warp_id * 16 * 4 + (group_id + 8) * 4 + lane_in_group;

  // if (offset1 > 32 * 4) {
  //   // printf("offset: %d\n", offset1);
  //   half * c_half_ptr = reinterpret_cast<half *>(c);
  //   printf("c : %f\n", __half2float(c_half_ptr[0]));
  // }

  C_ptr[offset1] = c[0];
  C_ptr[offset2] = c[1];
}

int main()
{

  half *d_C;
  half h_C[M * N];
  half h_CPU[M * N];
  half h_A[M * K];
  half h_B[K * N];

  fill_random(h_A, M, K);
  fill_random(h_B, K, N);

  for (int i = 0; i < M * N; i++)
  {
    h_C[i] = 0.0f;
  }

  half *d_A, *d_B;

  cudaMalloc((void **)&d_A, M * K * sizeof(half));
  cudaMalloc((void **)&d_B, K * N * sizeof(half));
  cudaMalloc((void **)&d_C, M * N * sizeof(half));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

  kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C);

  cuda_check_error();

  cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

  // print_matrix(h_C, M, N);

  CPU_gemm(h_A, h_B, h_CPU, M, N, K);

  compare_matrices(h_C, h_CPU, M, N);

  print_differnce(h_C, h_CPU, M, N, 0.0f);

  return 0;
}