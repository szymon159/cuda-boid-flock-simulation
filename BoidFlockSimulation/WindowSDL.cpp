#include "WindowSDL.h"

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
	_renderer = SDL_CreateRenderer(_window, -1, 0);
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

int WindowSDL::addBoidToRenderer(Boid boid, int boidSize)
{
	SDL_Rect dstrect = { boid.getX(), boid.getY(), boidSize, boidSize };

	if (SDL_RenderCopyEx(_renderer, _boidTexture, NULL, &dstrect, boid.getAngle(), NULL, SDL_FLIP_NONE) < 0)
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

// Private methods

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