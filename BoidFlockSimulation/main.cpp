#include "includes.h"
#include "window.h"

int main(int argc, char *argv[])
{
	SDL_Window *window = initWindow();
	if (window == nullptr)
		return EXIT_FAILURE;

	// Main window loop
	SDL_Event event;
	while (true)
	{
		while (SDL_PollEvent(&event) != 0)
		{
			if (event.type == SDL_QUIT)
			{
				destroyWindow(window);
				return EXIT_SUCCESS;
			}
		}
	}

	return EXIT_FAILURE;
}