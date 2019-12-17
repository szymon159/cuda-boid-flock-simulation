#pragma once
#include "includes.h"

#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"

void moveBoidKernelExecutor(float4 *&d_boids, 
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
							int windowWidth,
							int windowHeight,
							uint dt, 
							float boidSightRangeSquared);

void initializeCellsKernelExecutor (float4 *&d_boids,
									uint &boidArraySize,
									int *&d_boidId,
									int *&d_cellId,
									int *&d_cellBegin,
									int gridWidth,
									int cellSize,
									int cellCount);