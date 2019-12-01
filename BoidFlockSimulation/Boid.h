#pragma once

#include "includes.h"

class Boid
{
private:
	int _x;
	int _y;
	double _angle;

public: 
	Boid();
	Boid(int x, int y, double angle = 0.0);

	int getX();
	int getY();
	double getAngle();
};
