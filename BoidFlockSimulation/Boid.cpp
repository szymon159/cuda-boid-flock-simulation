#include "Boid.h"

Boid::Boid()
{

}

Boid::Boid(int windowWidth, int windowHeight, int size, float x, float y, float velocityX, float velocityY, float sightRange)
	:_windowWidth(windowWidth), _windowHeight(windowHeight), _size(size), _sightRange(sightRange), _sightRangeSquared(sightRange * sightRange)
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

float Boid::getAngle()
{
	return Calculator::getAngleFromVector(_velocity);
}

float Boid::getSightRangeSquared()
{
	return _sightRangeSquared;
}

void Boid::move()
{
	float2 movement = make_float2(0, 0);
	
	move(movement);
}

void Boid::move(float2 velocity)
{
	_velocity.x += velocity.x;
	_velocity.y += velocity.y;

	_position.x += _velocity.x;
	_position.y += _velocity.y;
}

void Boid::update(float4 newData)
{
	_position.x = newData.x;
	_position.y = newData.y;

	_velocity.x = newData.z;
	_velocity.y = newData.w;
}
