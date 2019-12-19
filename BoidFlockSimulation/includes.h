#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime> 

#include "SDL.h"
#include "SDL_image.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned int uint;

constexpr int WINDOW_HEIGHT = 1080;
constexpr int WINDOW_WIDTH = 1920;
constexpr int BOID_SIZE = 16;
constexpr float SIGHT_RANGE = 350;
constexpr int BOID_COUNT = 10000;

constexpr int BLOCK_COUNT = BOID_COUNT / 256 + (BOID_COUNT % 256 != 0);
constexpr float RADIAN_MULTIPLIER = (float)M_PI / 180;
constexpr float SIGHT_RANGE_SQUARED = SIGHT_RANGE * SIGHT_RANGE;