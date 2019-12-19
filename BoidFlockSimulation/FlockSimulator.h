#pragma once

#include "includes.h"

#include "WindowSDL.h"
#include "Boid.h"
#include "Calculator.h"

class FlockSimulator
{
private:
	//int _boidSize;
	//int _boidCount;
	//std::vector<Boid> _boids;
	/*float _boidSightRange;
	float _boidSightRangeSquared;*/

	float4 *h_boids;
	float4 *h_boidsDoubleBuffer;
	//int *h_boidId;
	//int *h_cellId;
	//int *h_cellIdDoubleBuffer;
	int *h_cellBegin;

	int h_boidId[BOID_COUNT];
	int h_cellId[BOID_COUNT];
	//int h_cellIdDoubleBuffer[BOID_COUNT];
	//int h_cellBegin;

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

	//float4 *getBoidsArray();
	//void updateBoidsPosition(float4 *boidsArray);

	int run(float *runTime, uint *framesGenerated);
	void update(uint dt);

	//void addBoid(float x, float y, float2 velocity);
	int drawBoids();
	void initializeCells();
	//void moveBoids(float dt);

private:
	void generateBoids();
};
