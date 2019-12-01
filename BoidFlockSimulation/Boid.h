#pragma once

#include "includes.h"

class Boid
{
private:
	int _x;
	int _y;
	int _angle;

public: 
	Boid();
	Boid(int x, int y, int angle);
};
