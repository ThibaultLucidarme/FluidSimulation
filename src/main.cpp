
#include <opencv/cv.hpp>//namespace
#include <opencv2/highgui/highgui.hpp>//imread

#include "CommandLineParser.hpp"
#include "ProgressBar.hpp"
#include "CPU.hpp"
#include "FluidSimulation.hpp"

using namespace std;
using namespace cv;

#define FORCE 100000.0f

void perturbateWithMouse(int event, int x, int y, int flags, void* userdata)
{
	static int x_init;
	static int y_init;
     if  ( event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN)
     {
          x_init = x;
          y_init = y;
     }
     else if  ( event == EVENT_LBUTTONUP )
     {
		  FluidSimulation* sim = static_cast<FluidSimulation*>(userdata);
          sim->AddDensity(x_init, y_init,255);
		  sim->AddVelocity(x_init, y_init, FORCE*(x-x_init)/100,FORCE*(y-y_init)/100 );
     }
     else if  ( event == EVENT_RBUTTONUP )
     {
		  FluidSimulation* sim = static_cast<FluidSimulation*>(userdata);
		  sim->AddVelocity(x_init, y_init, FORCE*(x-x_init)/100,FORCE*(y-y_init)/100 );
     }
}



int main( int argc, char** argv)
{
	
	string examples = " ******************** examples ***************************\n";
	examples = examples +" * fluidSim -v 0.01 -d 0.05 -p 50\n";
	examples = examples +" * fluidSim -v 0.005 -d 0.02 -p 100\n";
	examples = examples +" * fluidSim -v 0.001 -d 0.05\n";
	examples = examples +" *********************************************************\n\n";
	examples = examples +"Left click, drag and release to add density and perturbations\n";
	examples = examples +"Right click, drag and release to add perturbations without density\n";
	examples = examples +"Press 'q' or ESC to exit simulation\n";


	// Parse commandline for parameters
	p::CommandLineParser parser(argc, argv);
	int size = parser.addOption<int>("-n", 250, "Size of simulation domain");
	int maxIter = parser.addOption<int>("-i", -1, "Maximum number of simulation iteration (use <0 for infinite)");
	float diff = parser.addOption<float>("-d", 0.005, "Diffusion coefficient; Higher values make the fluid dissipate faster");
	float visc = parser.addOption<float>("-v", 0.001, "Viscosity coefficient; Higher values makes the fluid more compact and react better to external perturbations");
	int precision = parser.addOption<int>("-p", 20, "Precision of the simulation");
	float dt = parser.addOption<float>("-t", 0.001, "Temporal resolution of the simulation");
	bool initalPerturbation = (parser.addOption<int>("-z",1, "Add initial perturbation ( 0 | 1 )")==1);
	string initialDensity = parser.addOption<string>("--initial-density","", "use image as initial density distribution");
	parser.addHelpOption(examples);
	
	cout<<"Press 'q' or ESC to exit simulation"<<endl;
	int iter=0;
	initCPUcount();
	
	//Create a window
    namedWindow("fluid", 1);

	// Initialization
	FluidSimulation* sim = new FluidSimulation(size, diff, visc, precision, dt);
	setMouseCallback("fluid", perturbateWithMouse, sim);
	
	if(initialDensity!="")
	{
		Mat init_d = imread(initialDensity).t();
		sim->Initialize(init_d);
	}
	
	// Perturbation
	if(initalPerturbation)
	{
		if(initialDensity=="")
		{
			sim->AddDensity(size/2, 1,255);
		}
		
		sim->AddVelocity(size/2, 1, 0, FORCE*9.81 );
	}

		
	// Simulation
	ProgressBar pbar(&iter, &maxIter);
	do
	{		
		sim->Update();
		char key = sim->Display("fluid");		
		if( (key & 255) == 27 || key=='q' ) break;
		if( key=='r' ) sim->Initialize();

		pbar.Progress();
		
		iter++;
	} while (iter != maxIter);

	delete sim;
	
	return EXIT_SUCCESS;

}
