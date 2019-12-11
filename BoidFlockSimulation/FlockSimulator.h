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

	float4 *h_boids;
	__device__ float4 *d_boids;
	__device__ float4 *d_boidsDoubleBuffer;
	__device__ int *d_boidId;
	__device__ int *d_cellId;
	__device__ int *d_cellIdDoubleBuffer;
	__device__ int *d_cellBegin;
	size_t _boidArrSize;
	size_t _gridWidth;
	size_t _gridHeight;
	size_t _gridSize;

	WindowSDL *_window;

public:
	FlockSimulator(WindowSDL *window, int boidSize, float boidSightRange);
	~FlockSimulator();

	float4 *getBoidsArray();
	void updateBoidsPosition(float4 *boidsArray);

	int run();
	void update(float dt);

	void generateBoids(int count);
	void addBoid(float x, float y, float angle);
	int drawBoids();
	void moveBoids(float dt);
};
