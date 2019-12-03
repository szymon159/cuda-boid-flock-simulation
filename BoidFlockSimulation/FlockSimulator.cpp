#include "FlockSimulator.h"

FlockSimulator::FlockSimulator(WindowSDL *window, int boidSize)
	: _window(window), _boidSize(boidSize)
{

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

void FlockSimulator::moveBoids()
{
	size_t boidCount = _boids.size();
	for (size_t i = 0; i < boidCount; i++)
	{
		float2 separationVector;
		float alignmentFactor = 0.0;
		float2 cohesionVector;

		for (size_t j = 0; j < boidCount; j++)
		{
			if (i == j)
				continue;

			float distance = Calculator::calculateDistance(_boids[i].getCoordinates(), _boids[j].getCoordinates());

			Calculator::updateSeparationFactor(separationVector, _boids[i].getCoordinates(), _boids[j].getCoordinates(), distance);
			Calculator::updateAlignmentFactor(alignmentFactor, _boids[j].getAngle());
			Calculator::updateCohesionFactor(cohesionVector, _boids[j].getCoordinates());
		}
		boidCount--;

		Calculator::normalizeVector(separationVector);

		alignmentFactor = alignmentFactor / boidCount;
		float2 alignmentVector = Calculator::getVectorFromAngle(alignmentFactor);

		cohesionVector.x = cohesionVector.x / boidCount - _boids[i].getCoordinates().x;
		cohesionVector.y = cohesionVector.y / boidCount - _boids[i].getCoordinates().y;
		Calculator::normalizeVector(cohesionVector);
		
		float3 movement = Calculator::getMovementFromFactors(separationVector, alignmentVector, cohesionVector);

		_boids[i].move(movement);
	}
}
