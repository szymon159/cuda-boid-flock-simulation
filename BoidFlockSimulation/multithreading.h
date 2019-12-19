#pragma once
#include "includes.h"

#include "Threads.h"

void moveBoidCPU(float4 *&h_boids,
				float4 *&h_boidsDoubleBuffer,
				uint &arraySize,
				int *&h_boidId,
				int *&h_cellId,
				int *&h_cellIdDoubleBuffer,
				int *&h_cellBegin,
				int gridWidth,
				int gridHeight,
				int cellSize,
				int cellCount,
				uint dt);

void initializeCellsCPU(float4 *&h_boids,
						uint &boidArraySize,
						int *&h_boidId,
						int *&h_cellId,
						int *&h_cellBegin,
						int gridWidth,
						int cellSize,
						int cellCount);