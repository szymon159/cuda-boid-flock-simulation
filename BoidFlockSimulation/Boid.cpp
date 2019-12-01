#include "Boid.h"

Boid::Boid()
{

}

Boid::Boid(int x, int y, double angle)
	:_x(x), _y(y), _angle(angle)
{

}

int Boid::GetX()
{
	return _x;
}

int Boid::GetY()
{
	return _y;
}

double Boid::GetAngle()
{
	return _angle;
}