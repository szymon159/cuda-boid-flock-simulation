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

void Boid::move(float3 movement)
{
	float newX = _position.x + movement.x;
	//if (newX <= 0 || newX >= _windowWidth)
	//	return;

	float newY = _position.y + movement.y;
	//if (newY <= 0 || newY >= _windowHeight)
	//	return;
	
	_position.x = newX;
	_position.y = newY;
	_position.z = movement.z;
}