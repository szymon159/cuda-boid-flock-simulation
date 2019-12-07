#include "FlockSimulator.h"

#include "kernel.cuh"


// TODO: Add destructors
FlockSimulator::FlockSimulator(WindowSDL *window, int boidSize, float boidSightRange)
	: _window(window), _boidSize(boidSize), _boidSightRange(boidSightRange)
{
	_boidSightRangeSquared = _boidSightRange * _boidSightRange;
}

int FlockSimulator::run()
{
	// Main window loop
	SDL_Event event;
	float time = SDL_GetTicks();

	h_boids = getBoidsArray();
	
	_boidArrSize = sizeof(float4) * _boids.size();
	cudaMalloc((float4**)&d_boids, _boidArrSize);
	cudaMalloc((float4**)&d_boidsDoubleBuffer, _boidArrSize);

	while (true)
	{
		float dt = SDL_GetTicks() - time;
		time += dt;

		while (SDL_PollEvent(&event) != 0)
		{
			if (event.type == SDL_QUIT)
			{
				cudaFree(d_boids);
				cudaFree(d_boidsDoubleBuffer);
				free(h_boids);
				_window->destroyWindow();
				return 0;
			}
		}

		update(dt);

		if (drawBoids())
		{
			cudaFree(d_boids);
			cudaFree(d_boidsDoubleBuffer);
			free(h_boids);
			_window->destroyWindow();
			return 1;
		}
	}
}

void FlockSimulator::update(float dt)
{
	//// CPU
	//moveBoids(dt);
	


	// GPU
	cudaMemcpy(d_boids, h_boids, _boidArrSize, cudaMemcpyHostToDevice);

	boidMoveKernelExecutor(d_boids, d_boidsDoubleBuffer, _boidArrSize, dt, _boidSightRangeSquared);

	cudaMemcpy(h_boids, d_boids, _boidArrSize, cudaMemcpyDeviceToHost);

	updateBoidsPosition(h_boids);
}

float4 *FlockSimulator::getBoidsArray()
{
	float4 *result = (float4 *)malloc(_boids.size() * sizeof(float4));

	for (int i = 0; i < _boids.size(); i++)
	{
		result[i] = make_float4(
			_boids[i].getPosition().x,
			_boids[i].getPosition().y,
			_boids[i].getVelocity().x,
			_boids[i].getVelocity().y);
	}

	return result;
}

void FlockSimulator::updateBoidsPosition(float4 *boidsArray)
{
	int boidsCount = _boids.size();
	for (int i = 0; i < boidsCount; i++)
	{
		_boids[i].update(boidsArray[i]);
	}
}

void FlockSimulator::generateBoids(int count)
{
	int width = _window->getWidth();
	int height = _window->getHeight();

	for (int i = 0; i < count; i++)
	{
		addBoid(
			rand() % width,
			rand() % height,
			rand() % 360 - 179
		);
	}
}

void FlockSimulator::addBoid(float x, float y, float angle)
{
	float2 velocity = Calculator::getVectorFromAngle(angle);
	Boid newBoid(_window->getWidth(), _window->getHeight(), _boidSize, x, y, velocity.x, velocity.y);
	_boids.push_back(newBoid);
}

int FlockSimulator::drawBoids()
{
	if (_window->clearRenderer())
	{
		printf("Unable to clear renderer buffer! SDL_Error: %s \n", SDL_GetError());
		return 1;
	}

	size_t boidCount = _boids.size();
	for (size_t i = 0; i < boidCount; i++)
	{
		if (_window->addBoidToRenderer(_boids[i], _boidSize))
			return 1;
	}
	_window->render();

	return 0;
}

void FlockSimulator::moveBoids(float dt)
{
	size_t boidCount = _boids.size();
	float refreshRateCoeeficient = dt / 1000;

	for (size_t i = 0; i < boidCount; i++)
	{
		float2 separationVector;
		float2 alignmentVector;
		float2 cohesionVector;

		int boidsSeen = 0;

		for (size_t j = 0; j < boidCount; j++)
		{
			if (i == j)
				continue;

			float distance = Calculator::calculateDistance(_boids[i].getPosition(), _boids[j].getPosition());

			if (distance > _boidSightRangeSquared)
				continue;

			Calculator::updateSeparationFactor(separationVector, _boids[i].getPosition(), _boids[j].getPosition(), distance);
			Calculator::updateAlignmentFactor(alignmentVector, _boids[j].getVelocity());
			Calculator::updateCohesionFactor(cohesionVector, _boids[j].getPosition());

			boidsSeen++;
		}
		if (boidsSeen == 0)
		{
			_boids[i].move();
			continue;
		}

		separationVector.x = -separationVector.x;
		separationVector.y = -separationVector.x;
		Calculator::normalizeVector(separationVector);

		alignmentVector.x = 0.125 * alignmentVector.x / boidsSeen;
		alignmentVector.y = 0.125 * alignmentVector.y / boidsSeen;
		Calculator::normalizeVector(alignmentVector);

		cohesionVector.x = 0.001 * (cohesionVector.x / boidsSeen - _boids[i].getPosition().x);
		cohesionVector.y = 0.001 * (cohesionVector.y / boidsSeen - _boids[i].getPosition().y);
		Calculator::normalizeVector(cohesionVector);
		
		float2 movement = Calculator::getMovementFromFactors(separationVector, alignmentVector, cohesionVector, refreshRateCoeeficient);
		_boids[i].move(movement);
	}
}
