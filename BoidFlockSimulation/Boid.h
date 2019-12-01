#pragma once

#include "includes.h"

class Boid
{
private:
	int _x;
	int _y;
	double _angle;

	int _windowHeight;
	int _windowWidth;
	int _size;

public: 
	Boid();
	Boid(int windowWidth, int windowHeight, int size, int x, int y, double angle = 0.0);

	int getX();
	int getY();
	double getAngle();
	void move();
	void move(int newX, int newY);
	void move(int newX, int newY, double newAngle);
};
