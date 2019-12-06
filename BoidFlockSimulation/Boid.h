#pragma once

#include "includes.h"

#include "Calculator.h"

class Boid
{
private:
	float2 _position;
	float2 _velocity;

	int _size;
	float _sightRange;
	float _sightRangeSquared;

	int _windowHeight;
	int _windowWidth;

public: 
	Boid();
	Boid(int windowWidth, int windowHeight, int size, float x, float y, float velocity_x, float velocity_y, float sightRange);

	float2 getPosition();
	float2 getVelocity();
	int getX();
	int getY();
	float getAngle();
	float getSightRangeSquared();

	void move();
	void move(float2 velocity);
	void update(float4 newData);
};
