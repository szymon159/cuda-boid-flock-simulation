#pragma once

#include "includes.h"

#include "WindowSDL.h"
#include "Boid.h"

class FlockSimulator
{
private:
	int _boidSize;
	std::vector<Boid> _boids;

	WindowSDL *_window;

public:
	FlockSimulator(WindowSDL *window, int boidSize);

	void addBoid(int x, int y, int angle = 0);
	int drawBoids();
	void moveBoids();
};