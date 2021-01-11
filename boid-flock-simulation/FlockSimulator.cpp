#include "FlockSimulator.h"

#include "Threads.h"
#include "kernel.cuh"
#include "multithreading.h"


FlockSimulator::FlockSimulator(WindowSDL *window)
	: _window(window)
{	
	// Initialization of boids
	_boidArrSize = sizeof(float4) * BOID_COUNT;
	size_t boidCellArrSize = sizeof(int) * BOID_COUNT;

	h_boids = (float4 *)malloc(_boidArrSize);
	generateBoids();
	
	if (USE_GPU)
	{
		cudaMalloc((float4**)&d_boids, _boidArrSize);
		cudaMalloc((float4**)&d_boidsDoubleBuffer, _boidArrSize);
		cudaMemcpy(d_boids, h_boids, _boidArrSize, cudaMemcpyHostToDevice);
	}
	else
	{
		h_boidsDoubleBuffer = (float4 *)malloc(_boidArrSize);
	}

	// Initialization of grid
	_gridHeight = (uint)ceil(_window->getHeight() / SIGHT_RANGE);
	_gridWidth = (uint)ceil(_window->getWidth() / SIGHT_RANGE);
	_gridSize = _gridHeight * _gridWidth;

	if (USE_GPU)
	{
		cudaMalloc((int**)&d_boidId, boidCellArrSize);
		cudaMalloc((int**)&d_cellId, boidCellArrSize);
		cudaMalloc((int**)&d_cellIdDoubleBuffer, boidCellArrSize);
		cudaMalloc((int**)&d_cellBegin, sizeof(int) * _gridSize);
	}
	else
	{
		h_cellBegin = (int *)malloc(sizeof(int) * _gridSize);
	}

	initializeCells();
}

FlockSimulator::~FlockSimulator()
{
	if (USE_GPU)
	{
		cudaFree(d_boids);
		cudaFree(d_boidsDoubleBuffer);
		cudaFree(d_boidId);
		cudaFree(d_cellId);
		cudaFree(d_cellIdDoubleBuffer);
		cudaFree(d_cellBegin);
	}
	else
	{
		free(h_boidsDoubleBuffer);
		free(h_cellBegin);
	}

	free(h_boids);
	_window->destroyWindow();
}

int FlockSimulator::run(float *runTime, uint *framesGenerated)
{
	// Main window loop
	SDL_Event event;
	uint startTime = SDL_GetTicks();
	uint time = startTime;

	while (true)
	{
		uint dt = SDL_GetTicks() - time;
		time += dt;

		while (SDL_PollEvent(&event) != 0)
		{
			if (event.type == SDL_QUIT)
			{
				*runTime = (SDL_GetTicks() - startTime) / 1000.0f;

				_window->destroyWindow();
				return 0;
			}
		}

		update(dt);

		if (drawBoids())
		{
			_window->destroyWindow();
			return 1;
		}
		(*framesGenerated)++;
	}
}

// Trigger calculations
void FlockSimulator::update(uint dt)
{
	if (USE_GPU)
	{
		moveBoidGPU(d_boids, d_boidsDoubleBuffer, _boidArrSize, d_boidId, d_cellId, d_cellIdDoubleBuffer, d_cellBegin, _gridWidth, _gridHeight, (int)SIGHT_RANGE, _gridSize, dt);
		cudaMemcpy(h_boids, d_boids, _boidArrSize, cudaMemcpyDeviceToHost);
	}
	else
	{
		for (int i = 0; i < BOID_COUNT; i++)
		{
			Threads::moveBoidThreadWork(
				i, 
				h_boids,
				h_boids,
				h_boidId,
				h_cellId,
				h_cellId,
				h_cellBegin,
				_gridWidth,
				_gridHeight,
				(int)SIGHT_RANGE,
				dt);
		}

		thrust::sort_by_key(thrust::host, h_cellId, h_cellId + BOID_COUNT, h_boidId);
		memset(h_cellBegin, -1, _gridSize * sizeof(int));

		for (int i = 0; i < BOID_COUNT; i++)
		{
			Threads::updateCellsThreadWork(i, h_cellId, h_cellBegin, _gridSize);
		}
	}
}

void FlockSimulator::generateBoids()
{
	int width = _window->getWidth();
	int height = _window->getHeight();

	for (int i = 0; i < BOID_COUNT; i++)
	{
		h_boids[i] = {
			(float)(rand() % width),
			(float)(rand() % height),
			2.0f * rand() / (float)RAND_MAX - 1,
			2.0f * rand() / (float)RAND_MAX - 1
		};
	}
}

int FlockSimulator::drawBoids()
{
	if (_window->clearRenderer())
	{
		printf("Unable to clear renderer buffer! SDL_Error: %s \n", SDL_GetError());
		return 1;
	}

	for (size_t i = 0; i < BOID_COUNT; i++)
	{
		if (_window->addBoidToRenderer(h_boids[i]))
			return 1;
	}
	_window->render();

	return 0;
}

void FlockSimulator::initializeCells()
{
	if (USE_GPU)
	{
		initializeCellsGPU(d_boids, _boidArrSize, d_boidId, d_cellId, d_cellBegin, _gridWidth, (int)SIGHT_RANGE, _gridSize);
	}
	else
	{
		for (int i = 0; i < BOID_COUNT; i++)
		{
			Threads::initializeCellsThreadWork(i, h_boids, h_cellId, h_boidId, _gridWidth, (int)SIGHT_RANGE);
		}
		
		thrust::sort_by_key(thrust::host, h_cellId, h_cellId + BOID_COUNT, h_boidId);
		memset(h_cellBegin, -1, _gridSize * sizeof(int));

		for (int i = 0; i < BOID_COUNT; i++)
		{
			Threads::updateCellsThreadWork(i, h_cellId, h_cellBegin, _gridSize);
		}
	}

}