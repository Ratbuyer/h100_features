#include <stdio.h>

void cuda_check_error()
{
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

class cuda_timer
{

private:
  cudaEvent_t start, stop;

public:
  cuda_timer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~cuda_timer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void start_timer()
  {
    cudaEventRecord(start);
  }

  void stop_timer()
  {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  }

  float get_time()
  {
    float time;
    cudaEventElapsedTime(&time, start, stop);
    return time;
  }
};