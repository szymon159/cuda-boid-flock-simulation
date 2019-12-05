#include "FlockSimulator.h"

#include "kernel.h"

FlockSimulator::FlockSimulator(WindowSDL *window, int boidSize)
	: _window(window), _boidSize(boidSize)
{

}

int FlockSimulator::run()
{
	// Main window loop
	SDL_Event event;
	float time = SDL_GetTicks();

	while (true)
	{
		float dt = SDL_GetTicks() - time;
		time += dt;

		while (SDL_PollEvent(&event) != 0)
		{
			if (event.type == SDL_QUIT)
			{
				_window->destroyWindow();
				return 0;
			}
		}

		update(dt);

		if (drawBoids())
			return 1;
	}
}

void FlockSimulator::update(float dt)
{
	// CPU
	//moveBoids(dt);
	//

	// TODO: GPU
	// Mallocs etc
	float3 *h_boids = getBoidsArray();
	float3 item = h_boids[0];
	size_t size = sizeof(float3) * _boids.size();
	__device__ float3 *d_boids = 0;
	cudaMalloc((float3**)&d_boids, size);
	cudaMemcpy(d_boids, h_boids, size, cudaMemcpyHostToDevice);

	cudaSetDevice(0);
	moveKernelExecutor(d_boids, size, dt);
	//moveKernel <<< 1, 50 >>> (d_boids, size);

	cudaMemcpy(h_boids, d_boids, size);

	cudaFree(d_boids);
	free(h_boids);
	// Invoke kernel
	// Sync threads
	//

}

float3 *FlockSimulator::getBoidsArray()
{
	//Free it some day ;)
	float3 *result = (float3 *)malloc(_boids.size() * sizeof(float3));

	for (int i = 0; i < _boids.size(); i++)
	{
		result[i] = _boids[i].getPosition();
	}

	return result;
}

void FlockSimulator::generateBoids(int count, float sightRange)
{
	int width = _window->getWidth();
	int height = _window->getHeight();

	for (int i = 0; i < count; i++)
	{
		addBoid(
			rand() % width,
			rand() % height,
			rand() % 360 - 179,
			sightRange
		);
	}
}

void FlockSimulator::addBoid(float x, float y, float angle, float sightRange)
{
	float2 velocity = Calculator::getVectorFromAngle(angle);
	Boid newBoid(_window->getWidth(), _window->getHeight(), _boidSize, x, y, velocity.x, velocity.y, sightRange);
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

			if (distance > _boids[i].getSightRangeSquared())
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
