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

	return 0;
}

void WindowSDL::destroyWindow()
{
	SDL_DestroyWindow(_window);
	SDL_Quit();
	printf("Window destroyed successfully!\n");
}

