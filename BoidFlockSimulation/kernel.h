#pragma once
#include "includes.h"

void boidMoveKernelExecutor(float4 *&d_boids, size_t &arraySize, float dt);