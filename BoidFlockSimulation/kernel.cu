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

__device__ void updateSeparationFactor(float2 &separationFactor, const float2 &startBoidPosition, const float2 &targetBoidPosition, const float &distance)
{
	separationFactor.x += (startBoidPosition.x - targetBoidPosition.x);// / distance;
	separationFactor.y += (startBoidPosition.y - targetBoidPosition.y);// / distance;
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

__device__ float2 normalizeVector(float2 &vector)
{
	float length = vector.x * vector.x + vector.y * vector.y;
	length = sqrtf(length);

	vector.x /= length;
	vector.y /= length;

	return vector;
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
	return make_float2(boidData.w, boidData.x);
}

__device__ float2 getBoidVelocity(float4 boidData)
{
	return make_float2(boidData.y, boidData.z);
}

__device__ float4 getUpdatedBoidData(float4 oldBoidData, float2 movement = make_float2(0,0))
{
	float4 result;

	result.z = oldBoidData.z + movement.x;
	result.w = oldBoidData.w + movement.y;

	result.x = oldBoidData.x + result.z;
	result.y = oldBoidData.y + result.w;

	return result;
}

__global__ void boidMoveKernel(float4 *d_boids, size_t boidCount, float dt, float boidSightRangeSquared, int alreadyProcessedCount = 0)
{
	float refreshRateCoeeficient = dt / 1000;
	
	int idx = blockDim.x*blockIdx.x + threadIdx.x + alreadyProcessedCount;

	float2 boidPosition = getBoidPosition(d_boids[idx]);
	float2 boidVelocity = getBoidVelocity(d_boids[idx]);

	float2 separationVector;
	float2 alignmentVector;
	float2 cohesionVector;

	int boidsSeen = 0;

	float4 newBoidData;

	for (size_t j = 0; j < boidCount; j++)
	{
		if (idx == j)
			continue;

		float distance = calculateDistance(boidPosition, getBoidPosition(d_boids[j]));

		if (distance > boidSightRangeSquared)
			continue;

		updateSeparationFactor(separationVector, boidPosition, getBoidPosition(d_boids[j]), distance);
		updateAlignmentFactor(alignmentVector, getBoidVelocity(d_boids[j]));
		updateCohesionFactor(cohesionVector, getBoidPosition(d_boids[j]));

		boidsSeen++;
	}
	if (boidsSeen == 0)
	{
		newBoidData = getUpdatedBoidData(d_boids[idx]);
		return;
	}

	separationVector.x = -separationVector.x;
	separationVector.y = -separationVector.x;
	normalizeVector(separationVector);

	alignmentVector.x = 0.125 * alignmentVector.x / boidsSeen;
	alignmentVector.y = 0.125 * alignmentVector.y / boidsSeen;
	normalizeVector(alignmentVector);

	cohesionVector.x = 0.001 * (cohesionVector.x / boidsSeen - boidPosition.x);
	cohesionVector.y = 0.001 * (cohesionVector.y / boidsSeen - boidPosition.y);
	normalizeVector(cohesionVector);

	float2 movement = getMovementFromFactors(separationVector, alignmentVector, cohesionVector, refreshRateCoeeficient);

	newBoidData = getUpdatedBoidData(d_boids[idx], movement);

	//printf("Old: %f %f %f %f\n", d_boids[id].x, d_boids[id].y, d_boids[id].z, d_boids[id].w);
	//printf("New: %f %f %f %f\n\n", newBoidData.x, newBoidData.y, newBoidData.z, newBoidData.w);

	d_boids[idx] = newBoidData;
}

void boidMoveKernelExecutor(float4 *&d_boids, size_t &arraySize, float dt, float boidSightRangeSquared)
{
	size_t boidCount = arraySize / sizeof(float4);


	// TODO: do this threads number calculations only once
	int blockCount = boidCount >> 8;
	int threadsInLastBlockCount = boidCount % 256;
	int alreadyProcessedCount = boidCount - threadsInLastBlockCount;

	if(blockCount > 0)
		boidMoveKernel << <blockCount, 256 >> > (d_boids, boidCount, dt, boidSightRangeSquared);
	if (threadsInLastBlockCount > 0)
		boidMoveKernel << <1, threadsInLastBlockCount >> > (d_boids, boidCount, dt, boidSightRangeSquared, alreadyProcessedCount);
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
