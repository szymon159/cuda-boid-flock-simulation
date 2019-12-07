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
	float _boidSightRange;
	float _boidSightRangeSquared;

	WindowSDL *_window;

public:
	FlockSimulator(WindowSDL *window, int boidSize, float boidSightRange);

	float4 *getBoidsArray();
	void updateBoidsPosition(float4 *boidsArray);

	int run();
	void update(float dt);

	void generateBoids(int count);
	void addBoid(float x, float y, float angle);
	int drawBoids();
	void moveBoids(float dt);
};
