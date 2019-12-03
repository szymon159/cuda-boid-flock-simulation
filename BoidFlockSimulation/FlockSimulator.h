#pragma once

#include "includes.h"

#include "WindowSDL.h"
#include "Boid.h"
#include "Calculator.h"

class FlockSimulator
{
private:
	int _boidSize;
	std::vector<Boid> _boids;

	WindowSDL *_window;

public:
	FlockSimulator(WindowSDL *window, int boidSize);

	void addBoid(float x, float y, float angle = 0.0);
	int drawBoids();
	void moveBoids();
};
