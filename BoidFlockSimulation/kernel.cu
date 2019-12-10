#include "kernel.cuh"

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//void boidMoveKernelExecutor(float3 *&d_boids, size_t &arraySize, float dt)
//{
//	size_t boidCount = arraySize / sizeof(float3);
//
//	int blockCount = boidCount / 256;
//	int threadsInBlockCount;
//
//	boidMoveKernel << <1, 50 >> > (d_boids, boidCount, dt);
//}

__device__ float calculateDistance(float2 startPoint, float2 targetPoint)
{
	float distX = targetPoint.x - startPoint.x;
	distX *= distX;

	float distY = targetPoint.y - startPoint.y;
	distY *= distY;

	return distX + distY;
}

__device__ void updateSeparationFactor(float2 &separationFactor, const float2 &startBoidPosition, const float2 &targetBoidPosition)
{
	separationFactor.x += (startBoidPosition.x - targetBoidPosition.x);
	separationFactor.y += (startBoidPosition.y - targetBoidPosition.y);
}

__device__ void updateAlignmentFactor(float2 &alignmentFactor, const float2 &targetBoidVelocity)
{
	alignmentFactor.x += targetBoidVelocity.x;
	alignmentFactor.y += targetBoidVelocity.y;
}

__device__ void updateCohesionFactor(float2 &cohesionFactor, const float2 &targetBoidPosition)
{
	cohesionFactor.x += targetBoidPosition.x;
	cohesionFactor.y += targetBoidPosition.y;
}

__device__ float2 normalizeVector(const float2 &vector)
{
	float2 result = vector;
	if (fabs(vector.x) < 1e-8 || fabs(vector.y) < 1e-8)
	{
		result.x = vector.x * 1e8;
		result.y = vector.y * 1e8;
	}

	float length = result.x * result.x + result.y * result.y;
	length = sqrtf(length);

	if (isnan(result.x / length) || isnan(result.y / length))
	{
		return { sqrtf(2) / 2.0, sqrtf(2) / 2.0 };
	}

	return	{ result.x / length, result.y / length };
}

__device__ float2 getMovementFromFactors(float2 separationVector, float2 alignmentVector, float2 cohesionVector, float refreshRateCoefficient)
{
	float2 movement;
	//float angle;

	movement.x = refreshRateCoefficient * (separationVector.x + alignmentVector.x + cohesionVector.x);
	movement.y = refreshRateCoefficient * (separationVector.y + alignmentVector.y + cohesionVector.y);

	//if (movement.x != 0 || movement.y != 0)
	//	angle = getAngleFromVector(movement);
	//else
	//	angle = 0;

	//return make_float3(movement.x, movement.y, angle);

	return movement;
}

__device__ float2 getBoidPosition(float4 boidData)
{
	return make_float2(boidData.x, boidData.y);
}

__device__ float2 getBoidVelocity(float4 boidData)
{
	return make_float2(boidData.z, boidData.w);
}

__device__ float4 getUpdatedBoidData(float4 oldBoidData, float2 movement = { 0,0 })
{
	float4 result;

	result.z = oldBoidData.z + movement.x;
	result.w = oldBoidData.w + movement.y;

	result.x = oldBoidData.x + result.z;
	result.y = oldBoidData.y + result.w;

	return result;
}

__device__ int getCellId(float2 position, int gridWidth, int cellSize)
{
	int cellX = position.x / cellSize;
	int cellY = position.y / cellSize;

	return cellY * gridWidth + cellX;
}

__global__ void initializeCellsKernel ( float4 *d_boids,
										size_t boidCount,
										uint2 *d_boidCell,
										int gridWidth,
										int cellSize)
{
	int boidIdx = blockDim.x*blockIdx.x + threadIdx.x;
	if (boidIdx >= boidCount)
		return;

	float2 boidPosition = getBoidPosition(d_boids[boidIdx]);

	d_boidCell[boidIdx].x = getCellId(boidPosition, gridWidth, cellSize);
	d_boidCell[boidIdx].y = boidIdx;
}

__global__ void updateCellsBeginKernel (size_t boidCount,
										uint2* d_boidCell,
										int *d_cellBegin,
										int cellCount)
{
	int tId = blockDim.x*blockIdx.x + threadIdx.x;
	if (tId >= boidCount || tId == 0)
		return;

	if (d_boidCell[tId].x > cellCount)
		return;

	if (d_boidCell[tId - 1].x < d_boidCell[tId].x)
	{
		d_cellBegin[d_boidCell[tId].x] = tId;
	}
}


__global__ void moveBoidKernel (float4 *d_boids,
								float4 *d_boidsDoubleBuffer,
								size_t boidCount,
								uint2 *d_boidCell,
								uint2 *d_boidCellDoubleBuffer,
								int *d_cellBegin,
								float dt,
								float boidSightRangeSquared)
{
	int tId = blockDim.x*blockIdx.x + threadIdx.x;
	if (tId >= boidCount)
		return;

	float refreshRateCoeeficient = dt / 1000;
	int cellId = d_boidCell[tId].x;
	int boidIdx = d_boidCell[tId].y;

	float2 boidPosition = getBoidPosition(d_boids[boidIdx]);
	float2 boidVelocity = getBoidVelocity(d_boids[boidIdx]);

	float2 separationVector;
	float2 alignmentVector;
	float2 cohesionVector;

	int boidsSeen = 0;

	for (size_t j = 0; j < boidCount; j++)
	{
		if (boidIdx == j)
			continue;

		float distance = calculateDistance(boidPosition, getBoidPosition(d_boids[j]));

		if (distance > boidSightRangeSquared)
			continue;

		updateSeparationFactor(separationVector, boidPosition, getBoidPosition(d_boids[j]));
		updateAlignmentFactor(alignmentVector, getBoidVelocity(d_boids[j]));
		updateCohesionFactor(cohesionVector, getBoidPosition(d_boids[j]));

		boidsSeen++;
	}
	if (boidsSeen == 0)
	{
		d_boidsDoubleBuffer[boidIdx] = getUpdatedBoidData(d_boids[boidIdx]);
		return;
	}

	separationVector.x = -separationVector.x;
	separationVector.y = -separationVector.x;
	separationVector = normalizeVector(separationVector);

	alignmentVector.x = 0.125 * alignmentVector.x / boidsSeen;
	alignmentVector.y = 0.125 * alignmentVector.y / boidsSeen;
	alignmentVector = normalizeVector(alignmentVector);

	cohesionVector.x = 0.001 * (cohesionVector.x / boidsSeen - boidPosition.x);
	cohesionVector.y = 0.001 * (cohesionVector.y / boidsSeen - boidPosition.y);
	cohesionVector = normalizeVector(cohesionVector);

	float2 movement = getMovementFromFactors(separationVector, alignmentVector, cohesionVector, refreshRateCoeeficient);

	d_boidsDoubleBuffer[boidIdx] = getUpdatedBoidData(d_boids[boidIdx], movement);
}

void moveBoidKernelExecutor(float4 *&d_boids,
							float4 *&d_boidsDoubleBuffer,
							size_t &arraySize,
							uint2 *&d_boidCell,
							uint2 *&d_boidCellDoubleBuffer,
							int *&d_cellBegin,
							float dt,
							float boidSightRangeSquared)
{
	size_t boidCount = arraySize / sizeof(float4);

	// TODO: do this threads number calculations only once
	int blockCount = boidCount / 256;
	if (boidCount % 256 != 0)
	{
		blockCount++;
	}

	moveBoidKernel<<<blockCount, 256>>>(d_boids, d_boidsDoubleBuffer, boidCount, d_boidCell, d_boidCellDoubleBuffer, d_cellBegin, dt, boidSightRangeSquared);

	cudaThreadSynchronize();

	cudaMemcpy(d_boids, d_boidsDoubleBuffer, arraySize, cudaMemcpyDeviceToDevice);
	//printf("-------------------------------\n");
}

void initializeCellsKernelExecutor (float4 *&d_boids,
									size_t &boidArraySize,
									uint2 *&d_boidCell,
									int *&d_cellBegin,
									int gridWidth,
									int cellSize,
									int cellCount)
{
	size_t boidCount = boidArraySize / sizeof(float4);

	// TODO: do this threads number calculations only once
	int blockCount = boidCount / 256;
	if (boidCount % 256 != 0)
	{
		blockCount++;
	}

	initializeCellsKernel << <blockCount, 256 >> > (d_boids, boidCount, d_boidCell, gridWidth, cellSize);
	cudaThreadSynchronize();

	//sort

	cudaMemset(d_cellBegin, -1, cellCount);
	updateCellsBeginKernel << <blockCount, 256 >> > (boidCount, d_boidCell, d_cellBegin, cellCount);
	cudaThreadSynchronize();
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
