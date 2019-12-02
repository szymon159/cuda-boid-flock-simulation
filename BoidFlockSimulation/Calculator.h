#pragma once

#include "includes.h"

class Calculator
{
public:
	static void updateSeparationFactor(float2 &separationFactor, const float3 &startBoidPosition, const float3 &targetBoidPosition, const float &distance);
	static void updateAlignmentFactor(float &alignmentFactor, const float3 &targetBoidPosition);
	static void updateCohesionFactor(float2 &cohesionFactor, const float3 &targetBoidPosition);

	static float2 normalizeVector(float2 &vector);
	static float normalizeAngle(float &angle);
	static float2 getVectorFromAngle(float angle);
	static float getAngleFromVector(float2 vector);
};