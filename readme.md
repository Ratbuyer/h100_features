# Introduction

This repo contains demonstration codes on how to use Nvidia's H100 GPU's unique features via CUDA C++.

Each .cu file under `examples` demonstrate one specific feature of H100 and how to use it.

You should try to compile and run each file accoriding to the same order described in this readme in order to avoid confusion. You must have a CUDA 12.4 compiler and a sm90a GPU.

To compile sample code file, use make `file name`, refer to the makefile in this directory. For example, to compile tma_1d.cu, do `make tma_1d`. This will compile the file into a `bin` binary under the `bins` folder. To run the binary, do `make run`.

# Clusters

in sm90, we can optionaly group blocks in a grid into clusters, and blocks within a cluster can synchronize and communicate through a distributed shared memory. Refer to `cluster.cu` to see a sample code. Also check the cuda menu for details `https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters`

# Warp Group MMA

Warp Group MMA(wgmma) is a series of PTX instructions that has the common prefix: `wgmma.mma_async.sync.aligned` that performs matrix multiplication on certain sizes and data types. They are the advanced version of MMA instructions that is avaiable on GPUs of sm90+.

A warp group is a group of 4 warps, and this instruction require 4 warps to issue the instruction together, so this requires any kernel using wgmma having block size divisible by 4 warps.

Refer to the ptx menu `https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-operation-using-wgmma-mma-async-instruction` to see the available size and type combination. In `wgmma_dense.cu`, we demonstrate how to use it using `half` type for both A and B. The size we use is M=64, N = 8, K = 16. Try run the file by type `make dense` and do `make run`. The wgmma instruction is asynchronous, which means they will not block the execution of the kernel, and requires async proxy to ensure correctness. We need to use three instructions to ensure the correct behaviour. Refer to the menu for details - `https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-wgmma-proxy-operations`

- `wgmma.fence.sync.aligned;\n" ::: "memory"` - This instruction is required before any accumulator register in the wgmma instruction is accessed. The wgmma it self counts as accessing the accumulator so before every wgmma instruction we need to use this.

- `"wgmma.commit_group.sync.aligned;\n" ::: "memory"` - commit the issued wgmma instructions

- `"wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"` - wait for wgmma instruction to finish until `N` previous wgmma instructions are running. So if N=0 then this will wait for all previous wgmma to finish.

wgmma also has a sparse version, which is similar to mma.sp which requires a metadata instruction to pass the indeces of each 2:4 tile. Check the `wgmma_sparse.cu` file for sample.

Because wgmma is an async instruction, they can overlap with other computations like FMA in cuda core. Refer to `overlap.cu` to see how different interleaving instructions can overlap computations.

wgmma also supports a swizzle mode which requires us to load the matrices into shared memory into a different layout. However, in the context of our project we know A at compile time so we should rearrange it at compile time and flatten to a 1d array to maximize coalescing.

# TMA

Tensor memory accelerator is a H100 hardware that accelerates memory operations. We use TMA through a specific set of api that has the common prefix: `cp.async.bulk`. There are two sets of them, one is the regular `cp.async.bulk` set of instructions that deal with loading and storing continuous data to fixed data location. Another set is the `cp.async.bulk.tensor` instructions which allows the user to define a tensor in the global memory ranging from 1d to 5d and load and store a specific subtile of the tensor. So the 1d version of `cp.async.bulk.tensor` instructions covers the functionality of regular `cp.async.bulk.tensor`, we will only demo the advanced tensor version. The tensor version also helps saving register use and address generation because we only have to pass in coordinate of the subtile instead of an actual memory location.

The `cp.async.bulk.tensor` apis always requires an input tensor map which contains all the metadata of the tensor in the global memory. This includes the dimension, pointer to the start of the tensor, dimension of subtile, etc. Check `tma_tensor_map.cuh` to see the details of how to create such map.

In `tma_1d.cu` and `tma_2d.cu` we demonstrate how to use tma to load and store a subtile of an array(1d) and a matrix(2d). Read the codes and understand what each function call do. Notice load and store uses different synchronization methods - load uses a barrier whereas store uses commit and wait. Refer to the files under `logs` to see what is expected when running each file and try compile and run them yourself.

TMA instructions also support and advanced version of loading which is used combined with a clustered kernel, that allows one thread block to initiate the loading and let the rest of the thread blocks in the cluster to recieve a copy of the same data. This is called multicast, check multicast.cu to see how it works. Note that the multicast version of load has no built-in wrapper function so we have to insert the raw ptx code ourself. After reading the code you should see that the `ctaMask` register controls which thread blocks recieves the data. Try compiling and running the code with different `ctaMask` to better understand it's behaviour. This multicast functionality is very useful for matmul kernels as matmuls require a large amount of data reuse for both A and B.

In addition to multicast, TMA store also supports an advanced store and reduce version, which allows a few element wise reduction operation to take place with the source data and destination data and writing the result data to destination. Refer to `reduce_store.cu` to see example usage. Try using different reduction operations to test the behaviour. This functionality is potentially useful for matmul optimizations like k-cuts.