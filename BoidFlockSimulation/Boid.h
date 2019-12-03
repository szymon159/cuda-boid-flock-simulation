#pragma once

#include "includes.h"

class Boid
{
private:
	float3 _position;

	int _windowHeight;
	int _windowWidth;
	int _size;

public: 
	Boid();
	Boid(int windowWidth, int windowHeight, int size, float x, float y, float angle = 0.0);

	float3 getPosition();
	float2 getCoordinates();
	int getX();
	int getY();
	float getAngle();
	void move();
	void move(float newX, float newY);
	void move(float newX, float newY, float newAngle);
	void move(float3 newPosition);
};
