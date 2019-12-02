#include "Calculator.h"

void Calculator::updateSeparationFactor(float2 &separationFactor, const float3 &startBoidPosition, const float3 &targetBoidPosition, const float &distance)
{
	separationFactor.x += (startBoidPosition.x - targetBoidPosition.x) / distance;
	separationFactor.y += (startBoidPosition.y - targetBoidPosition.y) / distance;
}

void Calculator::updateAlignmentFactor(float &alignmentFactor, const float3 &targetBoidPosition)
{
	alignmentFactor += targetBoidPosition.z;
}

void Calculator::updateCohesionFactor(float2 &cohesionFactor, const float3 &targetBoidPosition)
{
	cohesionFactor.x += targetBoidPosition.x;
	cohesionFactor.y += targetBoidPosition.y;
}

float2 Calculator::normalizeVector(float2 &vector)
{
	float length = vector.x * vector.x + vector.y * vector.y;
	vector.x *= vector.x / length;
	vector.y *= vector.y / length;

	return vector;
}

float Calculator::normalizeAngle(float &angle)
{
	return (float)fmod(angle, 360);;
}

float2 Calculator::getVectorFromAngle(float angle)
{
	float2 result;
	float angleRad = angle * RADIAN_MULTIPLIER;

	result.x = sin(angleRad);
	result.y = cos(angleRad);
	
	return result;
}

float Calculator::getAngleFromVector(float2 vector)
{
	float angleRad = atanf(vector.x / vector.y);

	return angleRad / RADIAN_MULTIPLIER;
}
