#pragma once

#include "includes.h"

namespace Calculator
{
	__device__ __host__ static void updateSeparationFactor(float2 &separationFactor, const float2 &startBoidPosition, const float2 &targetBoidPosition)
	{
		separationFactor.x += (startBoidPosition.x - targetBoidPosition.x);
		separationFactor.y += (startBoidPosition.y - targetBoidPosition.y);
	}

	__device__ __host__ static void updateAlignmentFactor(float2 &alignmentFactor, const float2 &targetBoidVelocity)
	{
		alignmentFactor.x += targetBoidVelocity.x;
		alignmentFactor.y += targetBoidVelocity.y;
	}

	__device__ __host__ static void updateCohesionFactor(float2 &cohesionFactor, const float2 &targetBoidPosition)
	{
		cohesionFactor.x += targetBoidPosition.x;
		cohesionFactor.y += targetBoidPosition.y;
	}

	__device__ __host__ static float calculateDistance(float2 startPoint, float2 targetPoint)
	{
		float distX = targetPoint.x - startPoint.x;
		distX *= distX;

		float distY = targetPoint.y - startPoint.y;
		distY *= distY;

		return distX + distY;
	}

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

	__device__ __host__ static float2 getMovementFromFactors(float2 sumOfFactors, float refreshRateCoefficient)
	{
		float2 movement;

		movement.x = refreshRateCoefficient * sumOfFactors.x;
		movement.y = refreshRateCoefficient * sumOfFactors.y;

		return movement;
	}

	__device__ __host__ static float2 getBoidPosition(float4 boidData)
	{
		return make_float2(boidData.x, boidData.y);
	}

	__device__ __host__ static float2 getBoidVelocity(float4 boidData)
	{
		return make_float2(boidData.z, boidData.w);
	}

	__device__ __host__ static float4 getUpdatedBoidData(float4 oldBoidData, int windowWidth, int windowHeight, float2 movement = { 0,0 })
	{
		float4 result;

		result.z = oldBoidData.z + movement.x;
		result.w = oldBoidData.w + movement.y;
		if (result.z * result.z + result.w * result.w > 25)
		{
			result.z = oldBoidData.z;
			result.w = oldBoidData.w;
		}

		result.x = fmodf(oldBoidData.x + result.z, (float)windowWidth);
		if (result.x < 0)
			result.x += windowWidth;
		result.y = fmodf(oldBoidData.y + result.w, (float)windowHeight);
		if (result.y < 0)
			result.y += windowHeight;

		return result;
	}

	static float getAngleFromVector(float2 vector)
	{
		float2 normalized = normalizeVector(vector);
	
		float angleRad = acosf(normalized.x);

		float angle = angleRad / RADIAN_MULTIPLIER;
	
		if (normalized.y < 0)
			return -angle;
	
		return angle;
	}


	//static void updateSeparationFactor(float2 &separationFactor, const float2 &startBoidPosition, const float2 &targetBoidPosition, const float &distance);
	//static void updateAlignmentFactor(float2 &alignmentFactor, const float2 &targetBoidVelocity);
	//static void updateCohesionFactor(float2 &cohesionFactor, const float2 &targetBoidPosition);

	//static float2 normalizeVector(float2 &vector);
	//static float normalizeAngle(float &angle);
	//static float2 getVectorFromAngle(float angle);

	//static float calculateDistance(float2 startPoint, float2 targetPoint);
	//static float2 getMovementFromFactors(float2 separationVector, float2 alignmentVector, float2 cohesionVector, float refreshRateCoeficient);
};