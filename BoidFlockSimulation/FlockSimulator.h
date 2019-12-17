#pragma once

#include "includes.h"

#include "WindowSDL.h"
#include "Boid.h"
#include "Calculator.h"

class FlockSimulator
{
private:
	int _boidSize;
	int _boidCount;
	//std::vector<Boid> _boids;
	float _boidSightRange;
	float _boidSightRangeSquared;

	float4 *h_boids;
	__device__ float4 *d_boids;
	__device__ float4 *d_boidsDoubleBuffer;
	__device__ int *d_boidId;
	__device__ int *d_cellId;
	__device__ int *d_cellIdDoubleBuffer;
	__device__ int *d_cellBegin;
	uint _boidArrSize;
	uint _gridWidth;
	uint _gridHeight;
	uint _gridSize;

	WindowSDL *_window;

public:
	FlockSimulator(WindowSDL *window, int boidCount, int boidSize, float boidSightRange);
	~FlockSimulator();

	//float4 *getBoidsArray();
	//void updateBoidsPosition(float4 *boidsArray);

	int run();
	void update(uint dt);

	//void addBoid(float x, float y, float2 velocity);
	int drawBoids();
	//void moveBoids(float dt);

private:
	void generateBoids();
};
