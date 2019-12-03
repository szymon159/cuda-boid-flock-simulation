#include "Calculator.h"

void Calculator::updateSeparationFactor(float2 &separationFactor, const float2 &startBoidPosition, const float2 &targetBoidPosition, const float &distance)
{
	separationFactor.x += (startBoidPosition.x - targetBoidPosition.x) / distance;
	separationFactor.y += (startBoidPosition.y - targetBoidPosition.y) / distance;
}

void Calculator::updateAlignmentFactor(float &alignmentFactor, const float &targetBoidAngle)
{
	alignmentFactor += targetBoidAngle;
}

void Calculator::updateCohesionFactor(float2 &cohesionFactor, const float2 &targetBoidPosition)
{
	cohesionFactor.x += targetBoidPosition.x;
	cohesionFactor.y += targetBoidPosition.y;
}

float2 Calculator::normalizeVector(float2 &vector)
{
	float length = vector.x * vector.x + vector.y * vector.y;
	length = sqrtf(length);

	vector.x /= length;
	vector.y /= length;

	return vector;
}

float Calculator::normalizeAngle(float &angle)
{
	return (float)fmod(angle, 360);
}

float2 Calculator::getVectorFromAngle(float angle)
{
	float2 result;
	float angleRad = normalizeAngle(angle) * RADIAN_MULTIPLIER;

	result.x = sin(angleRad);
	result.y = cos(angleRad);
	
	return result;
}

float Calculator::getAngleFromVector(float2 vector)
{
	normalizeVector(vector);

	float angleRad = acosf(vector.x);
	float angle = angleRad / RADIAN_MULTIPLIER;

	if (vector.y < 0)
		return angle - 180;

	return angle;
}

float Calculator::calculateDistance(float2 startPoint, float2 targetPoint)
{
	float distX = targetPoint.x - startPoint.x;
	distX *= distX;

	float distY = targetPoint.y - startPoint.y;
	distY *= distY;

	return distX + distY;
}

float3 Calculator::getMovementFromFactors(float2 separationVector, float2 alignmentVector, float2 cohesionVector)
{
	float2 movement;
	float angle;
	movement.x = 0;
	movement.y = 0;
	
	movement.x = separationVector.x + alignmentVector.x + cohesionVector.x;
	movement.y = separationVector.y + alignmentVector.y + cohesionVector.y;

	if (movement.x != 0 || movement.y != 0)
		angle = getAngleFromVector(movement);
	else
		angle = 0;

	return make_float3(movement.x, movement.y, angle);
}
