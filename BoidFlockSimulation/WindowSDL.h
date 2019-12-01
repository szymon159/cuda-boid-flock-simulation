#pragma once

#include "includes.h"

#include "Boid.h"

class WindowSDL
{
private:
	int _backgroundColor[4];
	char *_windowTitle;
	int _windowHeight;
	int _windowWidth;
	std::vector<Boid> _boids;

	SDL_Window *_window;
	SDL_Renderer *_renderer;
	SDL_Texture *_boidTexture;

public:
	WindowSDL(int backgroundColor[4], char *windowTitle, int windowWidth, int windowHeight);

	int initWindow();
	void destroyWindow();
	int loadBoidTexture();

	void addBoidToWindow(int x, int y, int angle = 0);
	int drawBoids();
};

