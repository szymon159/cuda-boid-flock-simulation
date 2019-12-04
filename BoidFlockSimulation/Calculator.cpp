#include "Calculator.h"

void Calculator::updateSeparationFactor(float2 &separationFactor, const float2 &startBoidPosition, const float2 &targetBoidPosition, const float &distance)
{
	separationFactor.x += (startBoidPosition.x - targetBoidPosition.x);// / distance;
	separationFactor.y += (startBoidPosition.y - targetBoidPosition.y);// / distance;
}

void Calculator::updateAlignmentFactor(float2 &alignmentFactor, const float2 &targetBoidVelocity)
{
	alignmentFactor.x += targetBoidVelocity.x;
	alignmentFactor.y += targetBoidVelocity.y;
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
		return -angle;

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

float2 Calculator::getMovementFromFactors(float2 separationVector, float2 alignmentVector, float2 cohesionVector, float refreshRateCoefficient)
{
	float2 movement;
	//float angle;
	
	movement.x = refreshRateCoefficient * (separationVector.x + alignmentVector.x + cohesionVector.x);
	movement.y = refreshRateCoefficient * (separationVector.y + alignmentVector.y + cohesionVector.y);

	//if (movement.x != 0 || movement.y != 0)
	//	angle = getAngleFromVector(movement);
	//else
	//	angle = 0;

	//return make_float3(movement.x, movement.y, angle);

	return movement;
}
