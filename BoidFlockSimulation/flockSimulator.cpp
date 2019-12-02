#include "FlockSimulator.h"

FlockSimulator::FlockSimulator(WindowSDL *window, int boidSize)
	: _window(window), _boidSize(boidSize)
{

}

void FlockSimulator::addBoid(int x, int y, int angle)
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

void FlockSimulator::moveBoids()
{
	size_t boidCount = _boids.size();
	for (size_t i = 0; i < boidCount; i++)
	{
		_boids[i].move();
	}
}