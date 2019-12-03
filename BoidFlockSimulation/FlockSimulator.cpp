#include "FlockSimulator.h"

FlockSimulator::FlockSimulator(WindowSDL *window, int boidSize)
	: _window(window), _boidSize(boidSize)
{

}

int FlockSimulator::run()
{
	// Main window loop
	SDL_Event event;
	int time = SDL_GetTicks();
	int delay = 0;

	while (true)
	{
		while (SDL_PollEvent(&event) != 0)
		{
			if (event.type == SDL_QUIT)
			{
				_window->destroyWindow();
				return 0;
			}
		}

		delay = SDL_GetTicks() - time;
		printf("%d\n", delay);
		if (update(delay))
			return 1;
		time = SDL_GetTicks();

		if (drawBoids())
			return 1;
	}
}

int FlockSimulator::update(time_t delay)
{
	moveBoids(delay);

	return 0;
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
	Boid newBoid(_window->getWidth(), _window->getHeight(), _boidSize, x, y, angle);
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

void FlockSimulator::moveBoids(time_t delay)
{
	size_t boidCount = _boids.size();
	float refreshRateCoeeficient = (float)delay / 16.0;

	for (size_t i = 0; i < boidCount; i++)
	{
		float2 separationVector;
		float alignmentFactor = 0.0;
		float2 cohesionVector;

		int boidsSeen = 0;

		for (size_t j = 0; j < boidCount; j++)
		{
			if (i == j)
				continue;

			float distance = Calculator::calculateDistance(_boids[i].getCoordinates(), _boids[j].getCoordinates());

			if (distance > 40000)
				continue;

			Calculator::updateSeparationFactor(separationVector, _boids[i].getCoordinates(), _boids[j].getCoordinates(), distance);
			Calculator::updateAlignmentFactor(alignmentFactor, _boids[j].getAngle());
			Calculator::updateCohesionFactor(cohesionVector, _boids[j].getCoordinates());

			boidsSeen++;
		}
		if (boidsSeen == 0)
			continue;

		Calculator::normalizeVector(separationVector);

		alignmentFactor = alignmentFactor / boidsSeen;
		float2 alignmentVector = Calculator::getVectorFromAngle(alignmentFactor);

		cohesionVector.x = cohesionVector.x / boidsSeen - _boids[i].getCoordinates().x;
		cohesionVector.y = cohesionVector.y / boidsSeen - _boids[i].getCoordinates().y;
		Calculator::normalizeVector(cohesionVector);
		
		float3 movement = Calculator::getMovementFromFactors(separationVector, alignmentVector, cohesionVector, refreshRateCoeeficient);

		_boids[i].move(movement);
	}
}
