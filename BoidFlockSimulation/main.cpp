#include "includes.h"

#include "WindowSDL.h"
#include "FlockSimulator.h"

int BACKGROUND_COLOR[4] = { 0, 105, 148, 255 }; // Sea blue
char *WINDOW_TITLE = "Boid Flock Simulation";
int WINDOW_HEIGHT = 900;
int WINDOW_WIDTH = 900;
int BOID_SIZE = 60;

int main(int argc, char *argv[])
{
	srand((int)time(nullptr));

	WindowSDL window(BACKGROUND_COLOR, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);
	if (window.initWindow())
		return EXIT_FAILURE;

	FlockSimulator simulator(&window, BOID_SIZE);
	simulator.generateBoids(150);

	// Main window loop
	SDL_Event event;
	while (true)
	{
		while (SDL_PollEvent(&event) != 0)
		{
			if (event.type == SDL_QUIT)
			{
				window.destroyWindow();
				return EXIT_SUCCESS;
			}
		}

		SDL_Delay(50);
		simulator.moveBoids();
		simulator.drawBoids();
	}

	return EXIT_FAILURE;
}