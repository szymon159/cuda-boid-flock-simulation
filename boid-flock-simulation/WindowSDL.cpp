#include "WindowSDL.h"

#include "Calculator.h"

WindowSDL::WindowSDL(int backgroundColor[4], char *windowTitle, int windowWidth, int windowHeight)
	: _windowTitle(windowTitle), _windowWidth(windowWidth), _windowHeight(windowHeight), _window (nullptr), _renderer (nullptr)
{
	// Arrays in C++ ...
	for(int i = 0 ; i < 4; i++)
		_backgroundColor[i] = backgroundColor[i];
}

int WindowSDL::getWidth()
{
	return _windowWidth;
}

int WindowSDL::getHeight()
{
	return _windowHeight;
}

int WindowSDL::initWindow()
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
		return 1;
	}

	// Creating window
	_window = SDL_CreateWindow(_windowTitle, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _windowWidth, _windowHeight, SDL_WINDOW_SHOWN);
	if (_window == NULL)
	{
		printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
		return 1;
	}
	_renderer = SDL_CreateRenderer(_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (_renderer == NULL)
	{
		printf("Window renderer could not be created! SDL_Error: %s\n", SDL_GetError());
		destroyWindow();
		return 1;
	}
	SDL_SetRenderDrawColor(_renderer, _backgroundColor[0], _backgroundColor[1], _backgroundColor[2], _backgroundColor[3]);
	if (SDL_RenderClear(_renderer) < 0)
	{
		printf("Unable to clear renderer target! SDL_Error: %s \n", SDL_GetError());
		return 1;
	}
	SDL_RenderPresent(_renderer);
	printf("Window initialized successfully!\n");

	return loadBoidTexture();
}

void WindowSDL::destroyWindow()
{
	SDL_DestroyTexture(_boidTexture);
	SDL_DestroyRenderer(_renderer);
	SDL_DestroyWindow(_window);
	SDL_Quit();
	printf("Window destroyed successfully!\n");
}

int WindowSDL::addBoidToRenderer(float4 boid)
{
	float2 position = Calculator::getBoidPosition(boid);
	float angle = Calculator::getAngleFromVector(Calculator::getBoidVelocity(boid));

	SDL_Rect dstrect = { (int)position.x, (int)position.y, BOID_SIZE, BOID_SIZE };
	SDL_RendererFlip flip = (SDL_RendererFlip)(SDL_FLIP_NONE);

	if (SDL_RenderCopyEx(_renderer, _boidTexture, NULL, &dstrect, angle, NULL, flip) < 0)
	{
		printf("Unable to render boid! SDL_Error: %s \n", SDL_GetError());
		return 1;
	}

	return 0;
}

int WindowSDL::clearRenderer()
{
	if (SDL_RenderClear(_renderer) < 0)
	{
		printf("Unable to clear renderer target! SDL_Error: %s \n", SDL_GetError());
		return 1;
	}
	else
	{
		return 0;
	}
}

void WindowSDL::render()
{
	SDL_RenderPresent(_renderer);
}


int WindowSDL::loadBoidTexture()
{
	//Initialize PNG loading
	int imgFlags = IMG_INIT_PNG;
	if (!(IMG_Init(imgFlags) & imgFlags))
	{
		printf("SDL_image could not initialize! SDL_image Error: %s\n", IMG_GetError());
		return 1;
	}

	SDL_Surface *image = IMG_Load("boid.png");
	if (image == NULL)
	{
		printf("Unable to load image! SDL_image Error: %s\n", IMG_GetError());
		return 1;
	}

	_boidTexture = SDL_CreateTextureFromSurface(_renderer, image);
	SDL_FreeSurface(image);
	if (_boidTexture == NULL)
	{
		printf("Unable to create texture from image! SDL_image Error: %s \n", IMG_GetError());
		return 1;
	}

	return 0;
}