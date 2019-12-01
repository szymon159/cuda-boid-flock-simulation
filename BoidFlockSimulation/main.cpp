#include "includes.h"
#include "WindowSDL.h"

int BACKGROUND_COLOR[4] = { 0, 105, 148, 255 }; // Sea blue
char *WINDOW_TITLE = "Boid Flock Simulation";
int WINDOW_HEIGHT = 600;
int WINDOW_WIDTH = 600;
int BOID_SIZE = 30;

int main(int argc, char *argv[])
{
	WindowSDL window(BACKGROUND_COLOR, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, BOID_SIZE);

	if (window.initWindow())
		return EXIT_FAILURE;

	window.addBoidToWindow(20, 20, 225);
	window.addBoidToWindow(540, 20, -45);
	window.addBoidToWindow(20, 540, 135);
	window.addBoidToWindow(540, 540, 45);
	window.addBoidToWindow(290, 290, 0);

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

		SDL_Delay(15);
		window.moveBoids();
		window.drawBoids();
	}

	return EXIT_FAILURE;
}