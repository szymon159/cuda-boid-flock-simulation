#pragma once

#include "includes.h"

class Calculator
{
public:
	static void updateSeparationFactor(float2 &separationFactor, const float2 &startBoidPosition, const float2 &targetBoidPosition, const float &distance);
	static void updateAlignmentFactor(float &alignmentFactor, const float &targetBoidPosition);
	static void updateCohesionFactor(float2 &cohesionFactor, const float2 &targetBoidPosition);

	static float2 normalizeVector(float2 &vector);
	static float normalizeAngle(float &angle);
	static float2 getVectorFromAngle(float angle);
	static float getAngleFromVector(float2 vector);

	static float calculateDistance(float2 startPoint, float2 targetPoint);
	static float3 getMovementFromFactors(float2 separationVector, float2 alignmentVector, float2 cohesionVector, float refreshRateCoeficient);
};