#include "includes.h"

#include "WindowSDL.h"
#include "FlockSimulator.h"

int BACKGROUND_COLOR[4] = { 0, 105, 148, 255 }; // Sea blue
char *WINDOW_TITLE = "Boid Flock Simulation";

int main(int argc, char *argv[])
{
	srand((int)time(nullptr));

	WindowSDL window(BACKGROUND_COLOR, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);
	if (window.initWindow())
		return EXIT_FAILURE;

	FlockSimulator simulator(&window);

	float runTime;
	uint framesGenerated = 0;

	if (simulator.run(&runTime, &framesGenerated))
		return EXIT_FAILURE;

	printf("\nSimulator run for %.3f seconds and generated %u frames.\n", runTime, framesGenerated);
	printf("It gives average of %.1f FPS.\n", framesGenerated / (float)runTime);
	printf("If the result is close to refresh rate of your display, try increasing BOID_COUNT.\n");

	system("pause");
	return EXIT_SUCCESS;
}