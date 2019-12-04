#pragma once

#include "includes.h"

class Boid
{
private:
	float2 _position;
	float2 _velocity;

	int _windowHeight;
	int _windowWidth;
	int _size;

public: 
	Boid();
	Boid(int windowWidth, int windowHeight, int size, float x, float y, float velocity_x, float velocity_y);

	float2 getPosition();
	float2 getVelocity();
	int getX();
	int getY();

	void move(float2 velocity);
};
