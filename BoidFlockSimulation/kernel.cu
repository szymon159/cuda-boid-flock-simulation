#include "kernel.h"

#include "Calculator.h"

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


// TODO: Rename
__global__ void moveKernel(float3 *d_boids, size_t arraySize, float dt)
{
	size_t boidCount = arraySize / sizeof(float3);
	float refreshRateCoeeficient = dt / 50;

	float2 separationVector;
	float alignmentFactor = 0.0;
	float2 cohesionVector;

	int boidsSeen = 0;

	int i = threadIdx.x;

	for (size_t j = 0; j < boidCount; j++)
	{
		if (i == j)
			continue;

		float distance = Calculator::calculateDistance(make_float2(d_boids[i].x, d_boids[i].y), make_float2(d_boids[j].x, d_boids[j].y));

		if (distance > 10000)
			continue;

		Calculator::updateSeparationFactor(separationVector, make_float2(d_boids[i].x, d_boids[i].y), make_float2(d_boids[j].x, d_boids[j].y), distance);
		Calculator::updateAlignmentFactor(alignmentFactor, d_boids[j].z);
		Calculator::updateCohesionFactor(cohesionVector, make_float2(d_boids[j].x, d_boids[j].y));

		boidsSeen++;
	}
	if (boidsSeen == 0)
		return;

	Calculator::normalizeVector(separationVector);

	alignmentFactor = alignmentFactor / boidsSeen;
	float2 alignmentVector = Calculator::getVectorFromAngle(alignmentFactor);

	cohesionVector.x = cohesionVector.x / boidsSeen - d_boids[i].x;
	cohesionVector.y = cohesionVector.y / boidsSeen - d_boids[i].y;
	Calculator::normalizeVector(cohesionVector);

	float3 movement = Calculator::getMovementFromFactors(separationVector, alignmentVector, cohesionVector, refreshRateCoeeficient);

	d_boids[i] = movement;

	//float3 item = d_boids[0];
	//printf("Hello world: %f %f %f\n", item.x, item.y, item.z);
	//
}

void kernelWrapper(float3 *&d_boids, size_t &arraySize, float dt)
{
	moveKernel <<<1, 50>>>(d_boids, arraySize, dt);
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
