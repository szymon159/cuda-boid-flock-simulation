#pragma once

#include "includes.h"

class WindowSDL
{
private:
	int _backgroundColor[4];
	char *_windowTitle;
	int _windowHeight;
	int _windowWidth;
	std::vector<int> boids;

	SDL_Window *_window;
	SDL_Renderer *_renderer;

public:
	WindowSDL(int backgroundColor[4], char *windowTitle, int windowWidth, int windowHeight);

	int initWindow();
	void destroyWindow();
};

