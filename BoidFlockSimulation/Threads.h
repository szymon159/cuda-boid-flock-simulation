#pragma once

#include "includes.h"

#include "Calculator.h"
using namespace Calculator;

namespace Threads
{
	__device__ __host__ static void updateCellsThreadWork(int threadId, int *cellIds, int *cellBegins, int cellCount)
	{
		if (cellIds[threadId] < 0 || cellIds[threadId] >= cellCount)
			return;

		if (threadId == 0 || cellIds[threadId - 1] < cellIds[threadId])
		{
			cellBegins[cellIds[threadId]] = threadId;

			//printf("CellId: %d BeginId: %d\n", d_cellId[tId], d_cellBegin[d_cellId[tId]]);
		}
	}

	__device__ __host__ static void initializeCellsThreadWork(int threadId, float4 *boids, int *cellIds, int *boidIds, int gridWidth, int cellSize)
	{
		int boidIdx = threadId;

		float2 boidPosition = getBoidPosition(boids[boidIdx]);

		cellIds[boidIdx] = getCellId(boidPosition, gridWidth, cellSize);
		boidIds[boidIdx] = boidIdx;

		//printf("BoidId: %d, CellId: %d\n", d_boidId[boidIdx], d_cellId[boidIdx]);
	}

	__device__ __host__ static void moveBoidThreadWork(int threadId,
		float4 *boids,
		float4 *boidsDoubleBuffer,
		int *boidIds,
		int *cellIds,
		int *cellIdDoubleBuffer,
		int *cellBegins,
		int gridWidth,
		int gridHeight,
		int cellSize,
		uint dt)
	{
		float refreshRateCoeeficient = dt / 1000.0f;
		//printf("Refresh rate: %f\n", refreshRateCoeeficient);
		int cellId = cellIds[threadId];
		if (cellId < 0)
			return;

		int boidIdx = boidIds[threadId];

		float2 boidPosition = getBoidPosition(boids[boidIdx]);
		float2 boidVelocity = getBoidVelocity(boids[boidIdx]);

		float2 separationVector;
		float2 alignmentVector;
		float2 cohesionVector;

		int boidsSeen = 0;

		int neighCells[9];
		getNeighbourCells(cellId, gridWidth, gridHeight, neighCells);

		for (int i = 0; i < 9; i++)
		{
			int neighCellId = neighCells[i];
			float2 fakeBoidPosition = boidPosition;
			if (neighCellId < 0)
			{
				neighCellId *= (-1);
				fakeBoidPosition = getFakeBoidPosition(boidPosition, cellId, neighCellId, gridWidth, gridHeight);
			}

			int cellBegin = cellBegins[neighCellId];

			//printf("cellId: %d, cellbegin: %d\n", neighCellId, cellBegin);

			for (int j = cellBegin; j < BOID_COUNT; j++)
			{
				if (cellIds[j] != neighCellId)
					break;

				int targetBoidIdx = boidIds[j];
				if (boidIdx == targetBoidIdx)
					continue;

				float distance = calculateDistance(fakeBoidPosition, getBoidPosition(boids[targetBoidIdx]));

				if (distance > SIGHT_RANGE_SQUARED)
					continue;

				updateSeparationFactor(separationVector, fakeBoidPosition, getBoidPosition(boids[targetBoidIdx]));
				updateAlignmentFactor(alignmentVector, getBoidVelocity(boids[targetBoidIdx]));
				updateCohesionFactor(cohesionVector, getBoidPosition(boids[targetBoidIdx]));

				boidsSeen++;
			}
		}
		if (boidsSeen == 0)
		{
			boidsDoubleBuffer[boidIdx] = getUpdatedBoidData(boids[boidIdx]);
			cellIdDoubleBuffer[threadId] = cellId;
			return;
		}

		separationVector = { -separationVector.x, -separationVector.y };
		alignmentVector = { alignmentVector.x / boidsSeen, alignmentVector.y / boidsSeen };
		cohesionVector = { cohesionVector.x / boidsSeen - boidPosition.x, cohesionVector.y / boidsSeen - boidPosition.y };

		float2 vectors[] = { separationVector, alignmentVector, cohesionVector };
		float weights[] = { 1.0f, 0.125f, 0.01f };
		float2 sumOfFactors = addVectors(vectors, weights, 3);
		float2 movement = getMovementFromFactors(sumOfFactors, refreshRateCoeeficient);

		boidsDoubleBuffer[boidIdx] = getUpdatedBoidData(boids[boidIdx], movement);

		uint newCellId = getCellId(getBoidPosition(boidsDoubleBuffer[boidIdx]), gridWidth, cellSize);
		cellIdDoubleBuffer[threadId] = newCellId;
		//printf("cellId: %d, new: %d\n", cellId, newCellId);
	}
}