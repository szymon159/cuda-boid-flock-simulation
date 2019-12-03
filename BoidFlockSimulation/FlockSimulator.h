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

	int run();
	void update(float dt);

	void generateBoids(int count);

	void addBoid(float x, float y, float angle = 0.0);
	int drawBoids();
	void moveBoids(float dt);
};
