#pragma once

#include "includes.h"

class GPUCalculator
{
public:
	__device__ static void updateSeparationFactor(float2 &separationFactor, const float2 &startBoidPosition, const float2 &targetBoidPosition, const float &distance);
	__device__ static void updateAlignmentFactor(float2 &alignmentFactor, const float2 &targetBoidVelocity);
	__device__ static void updateCohesionFactor(float2 &cohesionFactor, const float2 &targetBoidPosition);

	__device__ static float2 normalizeVector(float2 &vector);
	__device__ static float normalizeAngle(float &angle);
	__device__ static float2 getVectorFromAngle(float angle);
	__device__ static float getAngleFromVector(float2 vector);

	__device__ static float calculateDistance(float2 startPoint, float2 targetPoint);
	__device__ static float2 getMovementFromFactors(float2 separationVector, float2 alignmentVector, float2 cohesionVector, float refreshRateCoeficient);
};