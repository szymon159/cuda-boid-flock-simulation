#include "WindowSDL.h"

WindowSDL::WindowSDL(int backgroundColor[4], char *windowTitle, int windowWidth, int windowHeight)
	: _windowTitle(windowTitle), _windowWidth(windowWidth), _windowHeight(windowHeight)
{
	// Arrays in C++ ...
	for(int i = 0 ; i < 4; i++)
		_backgroundColor[i] = backgroundColor[i];

	_window = nullptr;
	_renderer = nullptr;
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
	_renderer = SDL_CreateRenderer(_window, -1, 0);
	if (_renderer == NULL)
	{
		printf("Window renderer could not be created! SDL_Error: %s\n", SDL_GetError());
		destroyWindow();
		return 1;
	}
	SDL_SetRenderDrawColor(_renderer, _backgroundColor[0], _backgroundColor[1], _backgroundColor[2], _backgroundColor[3]);
	SDL_RenderClear(_renderer);
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
	if (_boidTexture == NULL)
	{
		printf("Unable to create texture from image! SDL_image Error: %s \n", IMG_GetError());
		return 1;
	}

	//
	SDL_Rect dstrect = { 5, 5, 30, 30 };
	SDL_RenderCopy(_renderer, _boidTexture, NULL, &dstrect);
	SDL_RenderPresent(_renderer);

	//

	return 0;
}

void WindowSDL::addBoidToWindow(int x, int y, int angle)
{
	Boid newBoid(x, y, angle);
	_boids.push_back(newBoid);
}