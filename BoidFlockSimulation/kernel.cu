#include "kernel.h"

#include "GPUCalculator.h"

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

__device__ float2 getBoidPosition(float4 boidData)
{
	return make_float2(boidData.w, boidData.x);
}

__device__ float2 getBoidVelocity(float4 boidData)
{
	return make_float2(boidData.y, boidData.z);
}

__global__ void boidMoveKernel(float4 *d_boids, size_t boidCount, float dt)
{
	float refreshRateCoeeficient = dt / 1000;
	
	//TODO: This is wrong index
	int id = threadIdx.x;

	float2 boidPosition = getBoidPosition(d_boids[id]);
	float2 boidVelocity = getBoidVelocity(d_boids[id]);

	float2 separationVector;
	float2 alignmentVector;
	float2 cohesionVector;

	int boidsSeen = 0;

	for (size_t j = 0; j < boidCount; j++)
	{
		if (id == j)
			continue;

		float distance = GPUCalculator::calculateDistance(boidPosition, getBoidPosition(d_boids[j]));

		// TODO: move sight range to flocksimulator class and then pass it to kernel
		if (distance > 10000)
			continue;

		GPUCalculator::updateSeparationFactor(separationVector, boidPosition, getBoidPosition(d_boids[j]), distance);
		GPUCalculator::updateAlignmentFactor(alignmentVector, getBoidVelocity(d_boids[j]));
		GPUCalculator::updateCohesionFactor(cohesionVector, getBoidPosition(d_boids[j]));

		boidsSeen++;
	}
	if (boidsSeen == 0)
	{
		//_boids[i].move();
		return;
	}

	separationVector.x = -separationVector.x;
	separationVector.y = -separationVector.x;
	GPUCalculator::normalizeVector(separationVector);

	alignmentVector.x = 0.125 * alignmentVector.x / boidsSeen;
	alignmentVector.y = 0.125 * alignmentVector.y / boidsSeen;
	GPUCalculator::normalizeVector(alignmentVector);

	cohesionVector.x = 0.001 * (cohesionVector.x / boidsSeen - boidPosition.x);
	cohesionVector.y = 0.001 * (cohesionVector.y / boidsSeen - boidPosition.y);
	GPUCalculator::normalizeVector(cohesionVector);

	float2 movement = GPUCalculator::getMovementFromFactors(separationVector, alignmentVector, cohesionVector, refreshRateCoeeficient);
	//_boids[i].move(movement);
}

void boidMoveKernelExecutor(float4 *&d_boids, size_t &arraySize, float dt)
{
	size_t boidCount = arraySize / sizeof(float4);

	int blockCount = boidCount / 256;
	int threadsInBlockCount;

	boidMoveKernel << <1, 50 >> > (d_boids, boidCount, dt);
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
