#include "multithreading.h"

// Async version of calculating updated boid's position on CPU
void moveBoidCPU(float4 *&h_boids, float4 *&h_boidsDoubleBuffer, uint &arraySize, int *&h_boidId, int *&h_cellId, int *&h_cellIdDoubleBuffer, int *&h_cellBegin, int gridWidth, int gridHeight, int cellSize, int cellCount, uint dt)
{
	std::thread threads[BOID_COUNT];
	for (int i = 0; i < BOID_COUNT; i++)
	{
		threads[i] = std::thread(
			Threads::moveBoidThreadWork,
				i,
				h_boids,
				h_boidsDoubleBuffer,
				h_boidId,
				h_cellId,
				h_cellIdDoubleBuffer,
				h_cellBegin,
				gridWidth,
				gridHeight,
				cellSize,
				dt);
	}
	for (int i = 0; i < BOID_COUNT; i++)
		threads[i].join();

	memcpy(h_cellId, h_cellIdDoubleBuffer, BOID_COUNT * sizeof(int));
	thrust::sort_by_key(thrust::host, h_cellId, h_cellId + BOID_COUNT, h_boidId);
	memset(h_cellBegin, -1, cellCount * sizeof(int));

	for (int i = 0; i < BOID_COUNT; i++)
		threads[i] = std::thread(Threads::updateCellsThreadWork, i, h_cellId, h_cellBegin, cellCount);
	for (int i = 0; i < BOID_COUNT; i++)
		threads[i].join();

	memcpy(h_boids, h_boidsDoubleBuffer, arraySize);
}

// Async version of initializing cells on CPU
void initializeCellsCPU(float4 *&h_boids, uint &boidArraySize, int *&h_boidId, int *&h_cellId, int *&h_cellBegin, int gridWidth, int cellSize, int cellCount)
{
	std::thread threads[BOID_COUNT];
	for (int i = 0; i < BOID_COUNT; i++)
		threads[i] = std::thread(Threads::initializeCellsThreadWork, i, h_boids, h_boidId, h_cellId, gridWidth, cellSize);
	for (int i = 0; i < BOID_COUNT; i++)
		threads[i].join();

	thrust::sort_by_key(thrust::host, h_cellId, h_cellId + BOID_COUNT, h_boidId);
	memset(h_cellBegin, -1, cellCount * sizeof(int));

	for (int i = 0; i < BOID_COUNT; i++)
		threads[i] = std::thread(Threads::updateCellsThreadWork, i, h_cellId, h_cellBegin, cellCount);
	for (int i = 0; i < BOID_COUNT; i++)
		threads[i].join();
}
