#pragma once
#include "includes.h"

// Invokes kernel calculating updated boid's position
void moveBoidGPU ( float4 *&d_boids,
					float4 *&d_boidsDoubleBuffer,
					uint &arraySize,
					int *&d_boidId,
					int *&d_cellId,
					int *&d_cellIdDoubleBuffer,
					int *&d_cellBegin,
					int gridWidth,
					int gridHeight,
					int cellSize,
					int cellCount,
					uint dt);


// Invokes kernel initializing cells
void initializeCellsGPU(float4 *&d_boids,
						uint &boidArraySize,
						int *&d_boidId,
						int *&d_cellId,
						int *&d_cellBegin,
						int gridWidth,
						int cellSize,
						int cellCount);