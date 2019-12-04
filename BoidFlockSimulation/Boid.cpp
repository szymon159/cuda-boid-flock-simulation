#include "Boid.h"

Boid::Boid()
{

}

Boid::Boid(int windowWidth, int windowHeight, int size, float x, float y, float velocityX, float velocityY)
	:_windowWidth(windowWidth), _windowHeight(windowHeight), _size(size)
{
	_position.x = x;
	_position.y = y;
	_velocity.x = velocityX;
	_velocity.y = velocityY;
}

float2 Boid::getPosition()
{
	return _position;
}

float2 Boid::getVelocity()
{
	return _velocity;
}

int Boid::getX()
{
	return (int)_position.x;
}

int Boid::getY()
{
	return (int)_position.y;
}

void Boid::move(float2 velocity)
{
	_velocity.x += velocity.x;
	_velocity.y += velocity.y;

	_position.x += _velocity.x;
	_position.y += _velocity.y;
}