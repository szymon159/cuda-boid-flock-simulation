#include "Boid.h"

Boid::Boid()
{

}

Boid::Boid(int windowWidth, int windowHeight, int size, float x, float y, float angle)
	:_windowWidth(windowWidth), _windowHeight(windowHeight), _size(size)
{
	_position.x = x;
	_position.y = y;
	_position.z = angle;

	srand((int)time(0));
}

float3 Boid::getPosition()
{
	return _position;
}

float2 Boid::getCoordinates()
{
	return make_float2(_position.x, _position.y);
}

int Boid::getX()
{
	return (int)_position.x;
}

int Boid::getY()
{
	return (int)_position.y;
}

float Boid::getAngle()
{
	return _position.z;
}

void Boid::move()
{
	float newX = _position.x + 5 + rand() % 11 - 5;
	float newY = _position.y + 5 + rand() % 11 - 5;
	float newAngle = _position.z + rand() % 11 - 5;

	move(newX, newY, newAngle);
}

void Boid::move(float newX, float newY)
{
	move(newX, newY, _position.z);
}

void Boid::move(float newX, float newY, float newAngle)
{
	if(newX > _size / 2 && newX < (_windowWidth - _size - 30))
		_position.x = newX;

	if (newY > _size / 2 && newY < (_windowHeight - _size - 30))
		_position.y = newY;

	_position.z = newAngle;
}