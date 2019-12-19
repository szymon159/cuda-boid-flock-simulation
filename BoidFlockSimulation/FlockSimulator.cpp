#include "FlockSimulator.h"

#include "kernel.cuh"


// TODO: Add destructors
FlockSimulator::FlockSimulator(WindowSDL *window, int boidCount, int boidSize, float boidSightRange)
	: _window(window), _boidCount(boidCount), _boidSize(boidSize), _boidSightRange(boidSightRange), _boidSightRangeSquared(boidSightRange * boidSightRange)
{	
	// Initialization of boids
	_boidArrSize = sizeof(float4) * boidCount;
	size_t boidCellArrSize = sizeof(int) * boidCount;

	h_boids = (float4 *)malloc(boidCount * sizeof(float4));
	cudaMalloc((float4**)&d_boids, _boidArrSize);
	cudaMalloc((float4**)&d_boidsDoubleBuffer, _boidArrSize);

	generateBoids();
	cudaMemcpy(d_boids, h_boids, _boidArrSize, cudaMemcpyHostToDevice);

	// Initialization of grid
	_gridHeight = (uint)ceil(_window->getHeight() / _boidSightRange);
	_gridWidth = (uint)ceil(_window->getWidth() / _boidSightRange);
	_gridSize = _gridHeight * _gridWidth;

	cudaMalloc((int**)&d_boidId, boidCellArrSize);
	cudaMalloc((int**)&d_cellId, boidCellArrSize);
	cudaMalloc((int**)&d_cellIdDoubleBuffer, boidCellArrSize);
	cudaMalloc((int**)&d_cellBegin, sizeof(int) * _gridSize);

	//TODO: Add a function which will do this on CPU/GPU
	initializeCellsKernelExecutor(d_boids, _boidArrSize, d_boidId, d_cellId, d_cellBegin, _gridWidth, (int)_boidSightRange, _gridSize);
}

FlockSimulator::~FlockSimulator()
{
	cudaFree(d_boids);
	cudaFree(d_boidsDoubleBuffer);
	cudaFree(d_boidId);
	cudaFree(d_cellId);	
	cudaFree(d_cellIdDoubleBuffer);
	cudaFree(d_cellBegin);
	free(h_boids);
	_window->destroyWindow();
}

int FlockSimulator::run()
{
	// Main window loop
	SDL_Event event;
	uint time = SDL_GetTicks();

	//h_boids = getBoidsArray();
	
	//_boidArrSize = sizeof(float4) * _boidCount;
	//size_t boidCellSize = sizeof(int) * _boidCount;

	//h_boids = (float4 *)malloc(_boidCount * sizeof(float4));
	//cudaMalloc((float4**)&d_boids, _boidArrSize);
	//cudaMalloc((float4**)&d_boidsDoubleBuffer, _boidArrSize);
	//cudaMalloc((int**)&d_boidId, boidCellSize);
	//cudaMalloc((int**)&d_cellId, boidCellSize);
	//cudaMalloc((int**)&d_cellIdDoubleBuffer, boidCellSize);
	//cudaMalloc((int**)&d_cellBegin, sizeof(int) * _gridSize);

	//cudaMemcpy(d_boids, h_boids, _boidArrSize, cudaMemcpyHostToDevice);

	// Initialize cells
	//initializeCellsKernelExecutor(d_boids, _boidArrSize, d_boidId, d_cellId, d_cellBegin, _gridWidth, (int)_boidSightRange, _gridSize);

	while (true)
	{
		uint dt = SDL_GetTicks() - time;
		time += dt;

		while (SDL_PollEvent(&event) != 0)
		{
			if (event.type == SDL_QUIT)
			{
				return 0;
			}
		}

		update(dt);

		if (drawBoids())
		{
			return 1;
		}
	}
}

void FlockSimulator::update(uint dt)
{
	//// CPU
	//moveBoids(dt);
	


	// GPU
	moveBoidKernelExecutor(d_boids, d_boidsDoubleBuffer, _boidArrSize, d_boidId, d_cellId, d_cellIdDoubleBuffer, d_cellBegin, _gridWidth, _gridHeight, (int)_boidSightRange, _gridSize, dt);
	cudaMemcpy(h_boids, d_boids, _boidArrSize, cudaMemcpyDeviceToHost);
	//updateBoidsPosition(h_boids);
}

//float4 *FlockSimulator::getBoidsArray()
//{
//	float4 *result = (float4 *)malloc(_boids.size() * sizeof(float4));
//
//	for (int i = 0; i < _boids.size(); i++)
//	{
//		result[i] = make_float4(
//			_boids[i].getPosition().x,
//			_boids[i].getPosition().y,
//			_boids[i].getVelocity().x,
//			_boids[i].getVelocity().y);
//	}
//
//	return result;
//}
//
//void FlockSimulator::updateBoidsPosition(float4 *boidsArray)
//{
//	//size_t boidsCount = _boids.size();
//	for (int i = 0; i < _boidsCount; i++)
//	{
//		_boids[i].update(boidsArray[i]);
//	}
//}

void FlockSimulator::generateBoids()
{
	int width = _window->getWidth();
	int height = _window->getHeight();

	for (int i = 0; i < _boidCount; i++)
	{
		h_boids[i] = {
			(float)(rand() % width),
			(float)(rand() % height),
			2.0f * rand() / (float)RAND_MAX - 1,
			2.0f * rand() / (float)RAND_MAX - 1
		};
	}
}

//void FlockSimulator::addBoid(float x, float y, float2 velocity)
//{
//	//float2 velocity = Calculator::getVectorFromAngle(angle);
//	Boid newBoid(_window->getWidth(), _window->getHeight(), _boidSize, x, y, velocity.x, velocity.y);
//	_boids.push_back(newBoid);
//}

int FlockSimulator::drawBoids()
{
	if (_window->clearRenderer())
	{
		printf("Unable to clear renderer buffer! SDL_Error: %s \n", SDL_GetError());
		return 1;
	}

	//size_t boidCount = _boids.size();
	for (size_t i = 0; i < _boidCount; i++)
	{
		if (_window->addBoidToRenderer(h_boids[i], _boidSize))
			return 1;
	}
	_window->render();

	return 0;
}
//
//void FlockSimulator::moveBoids(float dt)
//{
//	//size_t boidCount = _boids.size();
//	//float refreshRateCoeeficient = dt / 1000;
//
//	//for (size_t i = 0; i < boidCount; i++)
//	//{
//	//	float2 separationVector;
//	//	float2 alignmentVector;
//	//	float2 cohesionVector;
//
//	//	int boidsSeen = 0;
//
//	//	for (size_t j = 0; j < boidCount; j++)
//	//	{
//	//		if (i == j)
//	//			continue;
//
//	//		float distance = Calculator::calculateDistance(_boids[i].getPosition(), _boids[j].getPosition());
//
//	//		if (distance > _boidSightRangeSquared)
//	//			continue;
//
//	//		Calculator::updateSeparationFactor(separationVector, _boids[i].getPosition(), _boids[j].getPosition());
//	//		Calculator::updateAlignmentFactor(alignmentVector, _boids[j].getVelocity());
//	//		Calculator::updateCohesionFactor(cohesionVector, _boids[j].getPosition());
//
//	//		boidsSeen++;
//	//	}
//	//	if (boidsSeen == 0)
//	//	{
//	//		_boids[i].move();
//	//		continue;
//	//	}
//
//	//	separationVector.x = -separationVector.x;
//	//	separationVector.y = -separationVector.x;
//	//	Calculator::normalizeVector(separationVector);
//
//	//	alignmentVector.x = 0.125f * alignmentVector.x / boidsSeen;
//	//	alignmentVector.y = 0.125f * alignmentVector.y / boidsSeen;
//	//	Calculator::normalizeVector(alignmentVector);
//
//	//	cohesionVector.x = 0.001f * (cohesionVector.x / boidsSeen - _boids[i].getPosition().x);
//	//	cohesionVector.y = 0.001f * (cohesionVector.y / boidsSeen - _boids[i].getPosition().y);
//	//	Calculator::normalizeVector(cohesionVector);
//	//	
//	//	float2 movement = getMovementFromFactors(separationVector, alignmentVector, cohesionVector, refreshRateCoeeficient);
//	//	_boids[i].move(movement);
//	//}
//}
