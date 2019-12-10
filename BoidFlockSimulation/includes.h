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

constexpr float RADIAN_MULTIPLIER = (float)M_PI / 180;