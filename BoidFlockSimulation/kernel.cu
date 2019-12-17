#include "kernel.cuh"

#include "Calculator.h"

using namespace Calculator;

__device__ int getCellId(float2 position, int gridWidth, int cellSize)
{
	//printf("x: %f, y: %f, cellSize: %d\n", position.x, position.y, cellSize);
	int cellX = position.x / cellSize;
	int cellY = position.y / cellSize;

	return cellY * gridWidth + cellX;
}

__device__ void getNeighbourCells(int cellId, int gridWidth, int gridHeight, int (&neighbourCells)[9])
{
	int neighbourCellsCount = 0;

	int gridSize = gridWidth * gridHeight;
	int overflowMultiplier = 1;

	int centerCellId;
	for (int i = 0; i < 3; i++)
	{
		//Center of current row
		if (i == 0)
		{
			centerCellId = cellId;
		}
		else if (i == 1) //north
		{
			centerCellId = cellId - gridWidth;
			if (centerCellId < 0)
			{
				centerCellId += gridSize;
				overflowMultiplier = -1;
			}
		}
		else if (i == 2) //south
		{
			centerCellId = cellId + gridWidth;
			if (centerCellId >= gridSize)
			{
				centerCellId -= gridSize;
				overflowMultiplier = -1;
			}
		}

		neighbourCells[neighbourCellsCount++] = overflowMultiplier * centerCellId; //middle

		if (centerCellId % gridWidth != 0) //west
			neighbourCells[neighbourCellsCount++] = centerCellId - 1;
		else
			neighbourCells[neighbourCellsCount++] = -(centerCellId + gridWidth - 1);

		if ((centerCellId + 1) % gridWidth != 0) //east
			neighbourCells[neighbourCellsCount++] = centerCellId + 1;
		else
			neighbourCells[neighbourCellsCount++] = -(centerCellId - gridWidth + 1);

		overflowMultiplier = 1;
	}
}

__device__ float2 getFakeBoidPosition(float2 boidPosition, int cellId, int neighCellId, int gridWidth, int gridHeight, int windowWidth, int windowHeight)
{
	float2 result = boidPosition;

	int cellX = cellId % gridWidth;
	int cellY = cellId / gridWidth;

	int neighCellX = neighCellId % gridWidth;
	int neighCellY = neighCellId / gridWidth;


	if (cellX != neighCellX)
	{
		if (cellX == 0)
			result.x += windowWidth;
		else if (cellX == gridWidth - 1)
			result.x -= windowWidth;
	}

	if (cellY != neighCellY)
	{
		if (cellY == 0)
			result.x += windowHeight;
		else if (cellX == gridHeight - 1)
			result.x -= windowHeight;
	}

	return result;
}

__global__ void initializeCellsKernel ( float4 *d_boids,
										uint boidCount,
										int *d_boidId,
										int *d_cellId,
										int gridWidth,
										int cellSize)
{
	int boidIdx = blockDim.x*blockIdx.x + threadIdx.x;
	if (boidIdx >= boidCount)
		return;

	float2 boidPosition = getBoidPosition(d_boids[boidIdx]);

	d_cellId[boidIdx] = getCellId(boidPosition, gridWidth, cellSize);
	d_boidId[boidIdx] = boidIdx;

	//printf("BoidId: %d, CellId: %d\n", d_boidId[boidIdx], d_cellId[boidIdx]);
}

__global__ void updateCellsBeginKernel (uint boidCount,
										int *d_boidId,
										int *d_cellId,
										int *d_cellBegin,
										int cellCount)
{
	int tId = blockDim.x*blockIdx.x + threadIdx.x;
	if (tId >= boidCount)
		return;

	//printf("BoidId: %d, CellId: %d\n", d_boidId[tId], d_cellId[tId]);

	if (d_cellId[tId] < 0 || d_cellId[tId] > cellCount)
		return;

	if (tId == 0 || d_cellId[tId - 1] < d_cellId[tId])
	{
		d_cellBegin[d_cellId[tId]] = tId;

		//printf("CellId: %d BeginId: %d\n", d_cellId[tId], d_cellBegin[d_cellId[tId]]);
	}
}


__global__ void moveBoidKernel (float4 *d_boids,
								float4 *d_boidsDoubleBuffer,
								uint boidCount,
								int *d_boidId,
								int *d_cellId,
								int *d_cellIdDoubleBuffer,
								int *d_cellBegin,
								int gridWidth,
								int gridHeight,
								int cellSize,
								int windowWidth,
								int windowHeight,
								uint dt,
								float boidSightRangeSquared)
{
	int tId = blockDim.x*blockIdx.x + threadIdx.x;
	if (tId >= boidCount)
		return;

	float refreshRateCoeeficient = dt / 1000.0f;
	//printf("Refresh rate: %f\n", refreshRateCoeeficient);
	int cellId = d_cellId[tId];
	if (cellId < 0)
		return;

	int boidIdx = d_boidId[tId];

	float2 boidPosition = getBoidPosition(d_boids[boidIdx]);
	float2 boidVelocity = getBoidVelocity(d_boids[boidIdx]);

	float2 separationVector;
	float2 alignmentVector;
	float2 cohesionVector;

	int boidsSeen = 0;

	int neighCells[9];
	getNeighbourCells(cellId, gridWidth, gridHeight, neighCells);

	for (int i = 0; i < 9; i++)
	{
		int neighCellId = neighCells[i];
		float2 fakeBoidPosition = boidPosition;
		if (neighCellId < 0)
		{
			neighCellId *= (-1);
			fakeBoidPosition = getFakeBoidPosition(boidPosition, cellId, neighCellId, gridWidth, gridHeight, windowWidth, windowHeight);
		}

		int cellBegin = d_cellBegin[neighCellId];

		//printf("cellId: %d, cellbegin: %d\n", neighCellId, cellBegin);

		for (int j = cellBegin; j < boidCount; j++)
		{
			if (d_cellId[j] != neighCellId)
				break;

			int targetBoidIdx = d_boidId[j];
			if (boidIdx == targetBoidIdx)
				continue;

			float distance = calculateDistance(fakeBoidPosition, getBoidPosition(d_boids[targetBoidIdx]));

			if (distance > boidSightRangeSquared)
				continue;

			updateSeparationFactor(separationVector, fakeBoidPosition, getBoidPosition(d_boids[targetBoidIdx]));
			updateAlignmentFactor(alignmentVector, getBoidVelocity(d_boids[targetBoidIdx]));
			updateCohesionFactor(cohesionVector, getBoidPosition(d_boids[targetBoidIdx]));

			boidsSeen++;
		}
	}
	if (boidsSeen == 0)
	{
		d_boidsDoubleBuffer[boidIdx] = getUpdatedBoidData(d_boids[boidIdx], windowWidth, windowHeight);
		d_cellIdDoubleBuffer[tId] = cellId;
		return;
	}

	float2 sumOfFactors = { 0,0 };

	if (fabs(separationVector.x) > 1e-8 && fabs(separationVector.y) > 1e-8)
	{
		separationVector.x = -separationVector.x;
		separationVector.y = -separationVector.x;
		separationVector = normalizeVector(separationVector);

		sumOfFactors.x += separationVector.x;
		sumOfFactors.y += separationVector.y;
	}

	alignmentVector.x = alignmentVector.x / boidsSeen;
	alignmentVector.y = alignmentVector.y / boidsSeen;
	if (fabs(alignmentVector.x) > 1e-8 && fabs(alignmentVector.y) > 1e-8)
	{
		alignmentVector = normalizeVector(alignmentVector);

		sumOfFactors.x += 0.125 * alignmentVector.x;
		sumOfFactors.y += 0.125 * alignmentVector.y;
	}

	cohesionVector.x = (cohesionVector.x / boidsSeen - boidPosition.x);
	cohesionVector.y = (cohesionVector.y / boidsSeen - boidPosition.y);
	if (fabs(cohesionVector.x) > 1e-8 && fabs(cohesionVector.y) > 1e-8)
	{
		cohesionVector = normalizeVector(cohesionVector);

		sumOfFactors.x += 0.001 * cohesionVector.x;
		sumOfFactors.y += 0.001 * cohesionVector.y;
	}

	float2 movement = getMovementFromFactors(sumOfFactors, refreshRateCoeeficient);

	d_boidsDoubleBuffer[boidIdx] = getUpdatedBoidData(d_boids[boidIdx], windowWidth, windowHeight, movement);

	uint newCellId = getCellId(getBoidPosition(d_boidsDoubleBuffer[boidIdx]), gridWidth, cellSize);
	d_cellIdDoubleBuffer[tId] = newCellId;
	//printf("cellId: %d, new: %d\n", cellId, newCellId);
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
							int windowWidth,
							int windowHeight,
							uint dt,
							float boidSightRangeSquared)
{
	uint boidCount = arraySize / sizeof(float4);

	// TODO: do this threads number calculations only once
	int blockCount = boidCount / 256;
	if (boidCount % 256 != 0)
	{
		blockCount++;
	}

	moveBoidKernel<<<blockCount, 256>>>(d_boids, d_boidsDoubleBuffer, boidCount, d_boidId, d_cellId, d_cellIdDoubleBuffer, d_cellBegin, gridWidth, gridHeight, cellSize, windowWidth, windowHeight, dt, boidSightRangeSquared);
	cudaDeviceSynchronize();

	cudaMemcpy(d_cellId, d_cellIdDoubleBuffer, boidCount * sizeof(int), cudaMemcpyDeviceToDevice);
	//_sleep(15);
	thrust::sort_by_key(thrust::device_ptr<int>(d_cellId), thrust::device_ptr<int>(d_cellId + boidCount), thrust::device_ptr<int>(d_boidId));
	cudaMemset(d_cellBegin, -1, cellCount* sizeof(int));
	updateCellsBeginKernel << <blockCount, 256 >> > (boidCount, d_boidId, d_cellId, d_cellBegin, cellCount);
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
	uint boidCount = boidArraySize / sizeof(float4);

	// TODO: do this threads number calculations only once
	int blockCount = boidCount / 256;
	if (boidCount % 256 != 0)
	{
		blockCount++;
	}

	//printf("BEFORE:\n");

	initializeCellsKernel << <blockCount, 256 >> > (d_boids, boidCount, d_boidId, d_cellId, gridWidth, cellSize);
	cudaDeviceSynchronize();

	thrust::sort_by_key(thrust::device_ptr<int>(d_cellId), thrust::device_ptr<int>(d_cellId + boidCount), thrust::device_ptr<int>(d_boidId));

	//printf("AFTER:\n");

	cudaMemset(d_cellBegin, -1, cellCount * sizeof(int));
	updateCellsBeginKernel << <blockCount, 256 >> > (boidCount, d_boidId, d_cellId, d_cellBegin, cellCount);
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
