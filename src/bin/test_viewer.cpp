#include <stdlib.h>

#include "viewer_app.h"

int main(int argc, char **argv)
{
	ViewerApp viewer(argc, argv);
	viewer.start();

	return (EXIT_SUCCESS);
}
