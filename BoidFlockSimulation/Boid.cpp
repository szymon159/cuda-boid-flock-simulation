#include "Boid.h"

Boid::Boid()
{

}

Boid::Boid(int windowWidth, int windowHeight, int size, int x, int y, double angle)
	:_windowWidth(windowWidth), _windowHeight(windowHeight), _size(size), _x(x), _y(y), _angle(angle)
{
	srand((int)time(0));
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

void Boid::move()
{
	int newX = _x + 5 + rand() % 11 - 5;
	int newY = _y + 5 + rand() % 11 - 5;
	double newAngle = _angle + rand() % 11 - 5;

	move(newX, newY, newAngle);
}

void Boid::move(int newX, int newY)
{
	move(newX, newY, _angle);
}

void Boid::move(int newX, int newY, double newAngle)
{
	if(newX > _size / 2 && newX < (_windowWidth - _size - 30))
		_x = newX;

	if (newY > _size / 2 && newY < (_windowHeight - _size - 30))
		_y = newY;
	_angle = newAngle;
}