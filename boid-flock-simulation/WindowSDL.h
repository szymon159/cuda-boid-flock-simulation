#pragma once

#include "includes.h"

class WindowSDL
{
private:
	int _backgroundColor[4];
	char *_windowTitle;
	int _windowHeight;
	int _windowWidth;

	SDL_Window *_window;
	SDL_Renderer *_renderer;
	SDL_Texture *_boidTexture;

public:
	WindowSDL(int backgroundColor[4], char *windowTitle, int windowWidth, int windowHeight);

	int getWidth();
	int getHeight();

	int initWindow();
	void destroyWindow();

	int clearRenderer();
	int addBoidToRenderer(float4 boid);
	void render();

private:
	int loadBoidTexture();
};

