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

// Invokes kernel calculating updated boid's position
void moveBoidGPU(float4 *&d_boids,
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
}

// Invokes kernel initializing cells
void initializeCellsGPU (float4 *&d_boids,
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