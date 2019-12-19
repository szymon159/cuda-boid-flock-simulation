#include "kernel.cuh"

#include "Calculator.h"
#include "Threads.h"

using namespace Calculator;

__global__ void initializeCellsKernel ( float4 *d_boids,
										int *d_boidId,
										int *d_cellId,
										int gridWidth,
										int cellSize)
{
	int boidIdx = blockDim.x*blockIdx.x + threadIdx.x;
	if (boidIdx >= BOID_COUNT)
		return;

	Threads::initializeCellsThreadWork(boidIdx, d_boids, d_cellId, d_boidId, gridWidth, cellSize);
}

__global__ void updateCellsBeginKernel (int *d_boidId,
										int *d_cellId,
										int *d_cellBegin,
										int cellCount)
{
	int tId = blockDim.x*blockIdx.x + threadIdx.x;
	if (tId >= BOID_COUNT)
		return;

	Threads::updateCellsThreadWork(tId, d_cellId, d_cellBegin, cellCount);
}


__global__ void moveBoidKernel (float4 *d_boids,
								float4 *d_boidsDoubleBuffer,
								int *d_boidId,
								int *d_cellId,
								int *d_cellIdDoubleBuffer,
								int *d_cellBegin,
								int gridWidth,
								int gridHeight,
								int cellSize,
								uint dt)
{
	int tId = blockDim.x*blockIdx.x + threadIdx.x;
	if (tId >= BOID_COUNT)
		return;

	Threads::moveBoidThreadWork (tId,
						d_boids,
						d_boidsDoubleBuffer,
						d_boidId, 
						d_cellId, 
						d_cellIdDoubleBuffer, 
						d_cellBegin, 
						gridWidth, 
						gridHeight, 
						cellSize, 
						dt);
}

void moveBoidKernelExecutor(float4 *&d_boids,
							float4 *&d_boidsDoubleBuffer,
							uint &arraySize,
							int *&d_boidId,
							int *&d_cellId,
							int *&d_cellIdDoubleBuffer,
							int *&d_cellBegin,
							int gridWidth,
							int gridHeight,
							int cellSize,
							int cellCount,
							uint dt)
{
	moveBoidKernel<<<BLOCK_COUNT, 256>>>(d_boids, d_boidsDoubleBuffer, d_boidId, d_cellId, d_cellIdDoubleBuffer, d_cellBegin, gridWidth, gridHeight, cellSize, dt);
	cudaDeviceSynchronize();

	cudaMemcpy(d_cellId, d_cellIdDoubleBuffer, BOID_COUNT * sizeof(int), cudaMemcpyDeviceToDevice);
	thrust::sort_by_key(thrust::device_ptr<int>(d_cellId), thrust::device_ptr<int>(d_cellId + BOID_COUNT), thrust::device_ptr<int>(d_boidId));
	cudaMemset(d_cellBegin, -1, cellCount* sizeof(int));
	updateCellsBeginKernel << <BLOCK_COUNT, 256 >> > (d_boidId, d_cellId, d_cellBegin, cellCount);
	cudaDeviceSynchronize();

	cudaMemcpy(d_boids, d_boidsDoubleBuffer, arraySize, cudaMemcpyDeviceToDevice);
	//printf("-------------------------------\n");
}

void initializeCellsKernelExecutor (float4 *&d_boids,
									uint &boidArraySize,
									int *&d_boidId,
									int *&d_cellId,
									int *&d_cellBegin,
									int gridWidth,
									int cellSize,
									int cellCount)
{
	initializeCellsKernel << <BLOCK_COUNT, 256 >> > (d_boids, d_boidId, d_cellId, gridWidth, cellSize);
	cudaDeviceSynchronize();

	thrust::sort_by_key(thrust::device_ptr<int>(d_cellId), thrust::device_ptr<int>(d_cellId + BOID_COUNT), thrust::device_ptr<int>(d_boidId));

	cudaMemset(d_cellBegin, -1, cellCount * sizeof(int));
	updateCellsBeginKernel << <BLOCK_COUNT, 256 >> > (d_boidId, d_cellId, d_cellBegin, cellCount);
	cudaDeviceSynchronize();
}



//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
