#pragma once

#include "includes.h"

namespace Calculator
{
	// Updates value of separationFactor based on values of position of boid and its neighbour
	__device__ __host__ static void updateSeparationFactor(float2 &separationFactor, const float2 &startBoidPosition, const float2 &targetBoidPosition)
	{
		separationFactor.x += (startBoidPosition.x - targetBoidPosition.x);
		separationFactor.y += (startBoidPosition.y - targetBoidPosition.y);
	}

	// Updates value of alignmentFactor based on boid's neighbour velocity
	__device__ __host__ static void updateAlignmentFactor(float2 &alignmentFactor, const float2 &targetBoidVelocity)
	{
		alignmentFactor.x += targetBoidVelocity.x;
		alignmentFactor.y += targetBoidVelocity.y;
	}

	// Updates cohesionFactor based on boid's neighbour position
	__device__ __host__ static void updateCohesionFactor(float2 &cohesionFactor, const float2 &targetBoidPosition)
	{
		cohesionFactor.x += targetBoidPosition.x;
		cohesionFactor.y += targetBoidPosition.y;
	}

	// Returns distance between two points
	__device__ __host__ static float calculateDistance(float2 startPoint, float2 targetPoint)
	{
		float distX = targetPoint.x - startPoint.x;
		distX *= distX;

		float distY = targetPoint.y - startPoint.y;
		distY *= distY;

		return distX + distY;
	}

	// Normalizes vector
	__device__ __host__ static float2 normalizeVector(const float2 &vector)
	{
		float2 result = vector;

		float length = result.x * result.x + result.y * result.y;
		length = sqrtf(length);

		if (isnan(result.x / length) || isnan(result.y / length))
		{
			return { sqrtf(2) / 2.0f, sqrtf(2) / 2.0f };
		}

		return	{ result.x / length, result.y / length };
	}

	// Applies refreshRateCoeefficient to movement vector to make speed independent from refreshRate
	__device__ __host__ static float2 getMovementFromFactors(float2 sumOfFactors, float refreshRateCoefficient)
	{
		float2 movement;

		movement.x = refreshRateCoefficient * sumOfFactors.x;
		movement.y = refreshRateCoefficient * sumOfFactors.y;

		return movement;
	}

	// Returns position of boid from vector of position and velocity
	__device__ __host__ static float2 getBoidPosition(float4 boidData)
	{
		return make_float2(boidData.x, boidData.y);
	}

	// Returns velocity of boid from vector of position and velocity
	__device__ __host__ static float2 getBoidVelocity(float4 boidData)
	{
		return make_float2(boidData.z, boidData.w);
	}

	// Returns new position and velocity (after applying movement)
	__device__ __host__ static float4 getUpdatedBoidData(float4 oldBoidData, float2 movement = { 0,0 })
	{
		float4 result;

		result.z = oldBoidData.z + movement.x;
		result.w = oldBoidData.w + movement.y;
		if (result.z * result.z + result.w * result.w > MAX_SPEED)
		{
			result.z = oldBoidData.z;
			result.w = oldBoidData.w;
		}

		result.x = fmodf(oldBoidData.x + result.z, (float)WINDOW_WIDTH);
		if (result.x < 0)
			result.x += WINDOW_WIDTH;
		result.y = fmodf(oldBoidData.y + result.w, (float)WINDOW_HEIGHT);
		if (result.y < 0)
			result.y += WINDOW_HEIGHT;

		return result;
	}

	// Returns cellId from boid position
	__device__ __host__ static int getCellId(float2 position, int gridWidth, int cellSize)
	{
		int cellX = (int)position.x / cellSize;
		int cellY = (int)position.y / cellSize;

		return cellY * gridWidth + cellX;
	}

	// Updates neighbourCells[] with ids of cells who may contain boids reacting with boids from cellId
	__device__ __host__ static void getNeighbourCells(int cellId, int gridWidth, int gridHeight, int(&neighbourCells)[9])
	{
		int neighbourCellsCount = 0;

		int gridSize = gridWidth * gridHeight;
		int overflowMultiplier = 1;

		int centerCellId;
		for (int i = 0; i < 3; i++)
		{
			//Center of current row
			if (i == 0)
			{
				centerCellId = cellId;
			}
			else if (i == 1) //north
			{
				centerCellId = cellId - gridWidth;
				if (centerCellId < 0)
				{
					centerCellId += gridSize;
					overflowMultiplier = -1;
				}
			}
			else if (i == 2) //south
			{
				centerCellId = cellId + gridWidth;
				if (centerCellId >= gridSize)
				{
					centerCellId -= gridSize;
					overflowMultiplier = -1;
				}
			}

			neighbourCells[neighbourCellsCount++] = overflowMultiplier * centerCellId; //middle

			if (centerCellId % gridWidth != 0) //west
				neighbourCells[neighbourCellsCount++] = centerCellId - 1;
			else
				neighbourCells[neighbourCellsCount++] = -(centerCellId + gridWidth - 1);

			if ((centerCellId + 1) % gridWidth != 0) //east
				neighbourCells[neighbourCellsCount++] = centerCellId + 1;
			else
				neighbourCells[neighbourCellsCount++] = -(centerCellId - gridWidth + 1);

			overflowMultiplier = 1;
		}
	}

	// Returns weighted sum of normalized vectors. If values are really short, skip this vector to avoid problems with normalization
	__device__ __host__ static float2 addVectors(float2 vectors[], float weights[], int size)
	{
		float2 result = { 0,0 };

		for (int i = 0; i < size; i++)
		{
			if (fabs(vectors[i].x) > 1e-8 && fabs(vectors[i].y) > 1e-8)
			{
				float2 normalizedVector = normalizeVector(vectors[i]);

				result.x += weights[i] * normalizedVector.x;
				result.y += weights[i] * normalizedVector.y;
			}
		}

		return result;
	}

	// Returns position of boid out of the window in order to react with boids from opposite site of the screen
	__device__ __host__ static float2 getFakeBoidPosition(float2 boidPosition, int cellId, int neighCellId, int gridWidth, int gridHeight)
	{
		float2 result = boidPosition;

		int cellX = cellId % gridWidth;
		int cellY = cellId / gridWidth;

		int neighCellX = neighCellId % gridWidth;
		int neighCellY = neighCellId / gridWidth;


		if (cellX != neighCellX)
		{
			if (cellX == 0)
				result.x += WINDOW_WIDTH;
			else if (cellX == gridWidth - 1)
				result.x -= WINDOW_WIDTH;
		}

		if (cellY != neighCellY)
		{
			if (cellY == 0)
				result.x += WINDOW_HEIGHT;
			else if (cellX == gridHeight - 1)
				result.x -= WINDOW_HEIGHT;
		}

		return result;
	}

	// Return angle in degrees from vector (used for rotating boids's texture)
	static float getAngleFromVector(float2 vector)
	{
		float2 normalized = normalizeVector(vector);
	
		float angleRad = acosf(normalized.x);

		float angle = angleRad / RADIAN_MULTIPLIER;
	
		if (normalized.y < 0)
			return -angle;
	
		return angle;
	}
};