#include "Boid.h"

Boid::Boid()
{

}

Boid::Boid(int x, int y, double angle)
	:_x(x), _y(y), _angle(angle)
{

}

int Boid::getX()
{
	return _x;
}

int Boid::getY()
{
	return _y;
}

double Boid::getAngle()
{
	return _angle;
}