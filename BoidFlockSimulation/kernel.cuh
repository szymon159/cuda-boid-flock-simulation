#pragma once
#include "includes.h"

void moveBoidKernelExecutor(float4 *&d_boids, 
							float4 *&d_boidsDoubleBuffer, 
							size_t &arraySize, 
							uint2 *&d_boidCell,
							uint2 *&d_boidCellDoubleBuffer,
							int *&d_cellBegin,	
							float dt, 
							float boidSightRangeSquared);

void initializeCellsKernelExecutor (float4 *&d_boids,
									size_t &boidArraySize,
									uint2 *&d_boidCell,
									int *&d_cellBegin,
									int gridWidth,
									int cellSize,
									int cellCount);