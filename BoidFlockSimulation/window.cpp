#include "window.h"

int const BACKGROUND_COLOR[4] = { 0, 105, 148, 255 }; // Sea blue
char const *WINDOW_TITLE = "Boid Flock Simulation";
int const WINDOW_HEIGHT = 600;
int const WINDOW_WIDTH = 600;

SDL_Window *initWindow()
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
		return nullptr;
	}

	// Creating window
	SDL_Window *window = SDL_CreateWindow(WINDOW_TITLE, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
	if (window == NULL)
	{
		printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
		return nullptr;
	}
	SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);
	if (renderer == NULL)
	{
		printf("Window renderer could not be created! SDL_Error: %s\n", SDL_GetError());
		destroyWindow(window);
		return nullptr;
	}
	SDL_SetRenderDrawColor(renderer, BACKGROUND_COLOR[0], BACKGROUND_COLOR[1], BACKGROUND_COLOR[2], BACKGROUND_COLOR[3]);
	SDL_RenderClear(renderer);
	SDL_RenderPresent(renderer);
	printf("Window initialized successfully!\n");

	return window;
}

void destroyWindow(SDL_Window *window)
{
	SDL_DestroyWindow(window);
	SDL_Quit();
	printf("Window destroyed successfully!\n");
}