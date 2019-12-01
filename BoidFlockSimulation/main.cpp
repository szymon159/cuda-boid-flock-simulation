#include "includes.h"
#include "WindowSDL.h"

int BACKGROUND_COLOR[4] = { 0, 105, 148, 255 }; // Sea blue
char *WINDOW_TITLE = "Boid Flock Simulation";
int WINDOW_HEIGHT = 600;
int WINDOW_WIDTH = 600;

int main(int argc, char *argv[])
{
	WindowSDL window(BACKGROUND_COLOR, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);

	if (window.initWindow())
		return EXIT_FAILURE;

	int i = 10;
	// Main window loop
	SDL_Event event;
	while (true)
	{
		window.addBoidToWindow(i, i);
		i += 10;

		while (SDL_PollEvent(&event) != 0)
		{
			if (event.type == SDL_QUIT)
			{
				window.destroyWindow();
				return EXIT_SUCCESS;
			}
		}

		window.drawBoids();
	}

	return EXIT_FAILURE;
}