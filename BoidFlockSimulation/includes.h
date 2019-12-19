#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime> 
#include <thread>

#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"

#include "SDL.h"
#include "SDL_image.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned int uint;

constexpr bool USE_GPU = false;

constexpr int WINDOW_HEIGHT = 1080;
constexpr int WINDOW_WIDTH = 1920;
constexpr int BOID_SIZE = 16;
constexpr float SIGHT_RANGE = 500;
constexpr int BOID_COUNT = 5;
constexpr float MAX_SPEED = 9;

constexpr int BLOCK_COUNT = BOID_COUNT / 256 + (BOID_COUNT % 256 != 0);
constexpr float RADIAN_MULTIPLIER = (float)M_PI / 180;
constexpr float SIGHT_RANGE_SQUARED = SIGHT_RANGE * SIGHT_RANGE;