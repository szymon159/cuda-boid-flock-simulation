#pragma once

#include "includes.h"

#include "WindowSDL.h"
#include "Calculator.h"

class FlockSimulator
{
private:
	float4 *h_boids;
	float4 *h_boidsDoubleBuffer;
	int *h_cellBegin;

	int h_boidId[BOID_COUNT];
	int h_cellId[BOID_COUNT];

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
	FlockSimulator(WindowSDL *window);
	~FlockSimulator();

	int run(float *runTime, uint *framesGenerated);
	void update(uint dt);

	int drawBoids();
	void initializeCells();

private:
	void generateBoids();
};
