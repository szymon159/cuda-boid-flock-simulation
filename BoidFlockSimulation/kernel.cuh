#pragma once
#include "includes.h"

#include "thrust/sort.h"
#include "thrust/device_ptr.h"

void moveBoidKernelExecutor(float4 *&d_boids, 
							float4 *&d_boidsDoubleBuffer, 
							size_t &arraySize, 
							uint *&d_boidId,
							uint *&d_cellId,
							uint *&d_cellIdDoubleBuffer,
							int *&d_cellBegin,	
							float dt, 
							float boidSightRangeSquared);

void initializeCellsKernelExecutor (float4 *&d_boids,
									size_t &boidArraySize,
									uint *&d_boidId,
									uint *&d_cellId,
									int *&d_cellBegin,
									int gridWidth,
									int cellSize,
									int cellCount);