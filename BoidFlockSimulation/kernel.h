#pragma once
#include "includes.h"

void kernelWrapper(float3 *&d_boids, size_t &arraySize);
//__global__ void moveKernel(float3 *d_boids, size_t arraySize);