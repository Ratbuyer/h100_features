#include <stdio.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <random>
#include <map>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                              \
  {                                                                                      \
    cudaError_t error_code = callstr;                                                    \
    if (error_code != cudaSuccess)                                                       \
    {                                                                                    \
      std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
      assert(0);                                                                         \
    }                                                                                    \
  }
#endif

half rand_half()
{
  return __float2half(5.0f * rand() / RAND_MAX);
}

int rand_int(int max)
{
  return rand() % max;
}

void print_matrix(half *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%f ", __half2float(matrix[i * cols + j]));
    }
    printf("\n");
  }
  printf("\n");
}

void print_matrix(int *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%d ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void fill_random(half *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      float value = 0.1f * rand() / RAND_MAX;
      matrix[i * cols + j] = __float2half(value);
    }
  }
}

void fill_fixed(half *matrix, int rows, int cols, float value)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      matrix[i * cols + j] = __float2half(value);
    }
  }
}

void CPU_gemm(half *A, half *B, half *C, int M, int N, int K)
{
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      C[i * N + j] = 0;
      for (int k = 0; k < K; k++)
      {
        float a = __half2float(A[i * K + k]);
        float b = __half2float(B[k * N + j]);
        float c = __half2float(C[i * N + j]);
        float new_c = a * b + c;
        C[i * N + j] = __float2half(new_c);
      }
    }
  }
}

void compare_matrices(half *A, half *B, int rows, int cols)
{
  float total_diff = 0.0;
  int total_value = 0;

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      float a = __half2float(A[i * cols + j]);
      float b = __half2float(B[i * cols + j]);
      total_diff += fabs((a - b) / a);
      total_value += a / 2;
    }
  }

  float percentage_diff = (total_diff / total_value) * 100;
  printf("Total error: %.2f%%\n", percentage_diff);
}

void print_differnce(half *A, half *B, int rows, int cols, float error_range)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      float a = __half2float(A[i * cols + j]);
      float b = __half2float(B[i * cols + j]);
      bool is_same = a - error_range < b && a + error_range > b;
      if (!is_same)
      {
        printf("Error at (%d, %d) : %f != %f\n", i, j, a, b);
      }
    }
  }
}

void compress24(half *dense, half *sparse, int rows, int cols)
{
  assert(rows * cols % 4 == 0);

  memset(sparse, 0, rows * cols / 2 * sizeof(half));

  int counter;

  for (int i = 0; i < rows * cols; i += 4)
  {
    int sparse_offset = i / 2;

    counter = 0;

    for (int j = 0; j < 4; j++)
    {
      float value = __half2float(dense[i + j]);
      if (value != 0)
      {
        assert(counter < 2);
        sparse[sparse_offset + counter] = dense[i + j];
        counter++;
      }
    }
  }
}

void fill_24(half *matrix, int rows, int cols)
{
  assert(rows * cols % 4 == 0);

  for (int i = 0; i < rows * cols; i += 4)
  {
    matrix[i] = 0.0;
    matrix[i + 1] = 0.0;
    matrix[i + 2] = 0.0;
    matrix[i + 3] = 0.0;

    int position1 = rand() % 4;
    int position2 = rand() % 4;

    // position2 = position2 == position1 ? (position2 + 1) % 4 : position2;

    // matrix[i + position1] = __float2half(1.0f);
    // matrix[i + position2] = __float2half(1.0f);

    matrix[i + position1] = __float2half(rand_half());
    matrix[i + position2] = __float2half(rand_half());
  }
}

__host__ int inspect_metadata(half *mat, u_int32_t *meta, int M, int K)
{
  std::map<std::string, int> metaMap;

  metaMap["1100"] = 0x4;
  metaMap["1010"] = 0x8;
  metaMap["1001"] = 0xC;
  metaMap["0110"] = 0x9;
  metaMap["0101"] = 0xD;
  metaMap["0011"] = 0xE;

  metaMap["1000"] = 0x0;
  metaMap["0100"] = 0x1;
  metaMap["0010"] = 0x2;
  metaMap["0001"] = 0x3;

  metaMap["0000"] = 0xF;

  const int total_size = (M / 16) * (K / 16);

  meta = new u_int32_t[total_size * 8];

  int zero_tile = 0;

  for (int m = 0; m < M / 16; m++)
  {
    for (int k = 0; k < K / 16; k++)
    {
      for (int m2 = 0; m2 < 8; m2++)
      {
        unsigned int metadata = 0;
        for (int k2 = 0; k2 < 4; k2++)
        {
          std::string key = "";
          int counter = 0;
          for (int i = 0; i < 4; i++)
          {
            int index = (m * 16 + m2) * K + k * 16 + k2 * 4 + i;
            float value = __half2float(mat[index]);

            if (value != 0.0f)
            {
              key += "1";
              counter++;
            }
            else
            {
              key += "0";
            }
          }

          metadata |= metaMap[key] << (k2 * 4);

          if (counter == 0)
          {
            zero_tile++;
          }
        }
        for (int k2 = 0; k2 < 4; k2++)
        {
          std::string key = "";
          int counter = 0;
          for (int i = 0; i < 4; i++)
          {
            int index = (m * 16 + m2 + 8) * K + k * 16 + k2 * 4 + i;
            float value = __half2float(mat[index]);

            if (value != 0.0f)
            {
              key += "1";
              counter++;
            }
            else
            {
              key += "0";
            }
          }

          metadata |= metaMap[key] << (k2 * 4 + 16);

          if (counter == 0)
          {
            zero_tile++;
          }
        }
        int blockId = m * K / 16 + k;

        meta[blockId * 8 + m2] = metadata;
      }
    }
  }

  printf("zero tile: %d\n", zero_tile);

  double persentage = (double)zero_tile / (double)(M * K / 4) * 100.0;

  printf("zero tile persentage: %lf\n", persentage);

  return total_size * 8;
}

void wgmma_reorder_A(half *A, half *reordered_A, int M, int K, int M2, int K2)
{

  int blocks_in_m = M / M2;
  int blocks_in_k = K / K2;

  for (int m = 0; m < blocks_in_m; m++)
  {
    for (int k = 0; k < blocks_in_k; k++)
    {
      for (int i = 0; i < M2; i++)
      {
        for (int j = 0; j < K2; j++)
        {
        }
      }
    }
  }
}