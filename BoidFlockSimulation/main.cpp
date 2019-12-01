#include "includes.h"

const int BACKGROUND_COLOR[4] = { 0, 105, 148, 255 }; // Sea blue
const char *WINDOW_TITLE = "Boid Flock Simulation";
const int WINDOW_HEIGHT = 600;
const int WINDOW_WIDTH = 600;

int main(int argc, char *argv[])
{
	SDL_Init(SDL_INIT_VIDEO);

	// Creating window
	SDL_Window *window = SDL_CreateWindow(WINDOW_TITLE, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
	if (window == NULL)
	{
		printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
		return 1;
	}
	SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);
	if (renderer == NULL)
	{
		printf("Window renderer could not be created! SDL_Error: %s\n", SDL_GetError());
		return 1;
	}
	SDL_SetRenderDrawColor(renderer, BACKGROUND_COLOR[0], BACKGROUND_COLOR[1], BACKGROUND_COLOR[2], BACKGROUND_COLOR[3]);
	SDL_RenderClear(renderer);
	SDL_RenderPresent(renderer);
	printf("Window initialized successfully\n");

	// Main window loop
	SDL_Event event;
	while (true)
	{
		while (SDL_PollEvent(&event) != 0)
		{
			if (event.type == SDL_QUIT)
			{
				return 0;
			}
		}
	}

	return 0;
}