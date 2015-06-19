#include "FluidSimulation.hpp"
#include <cstdlib>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>//imshow
#include <thread>

#define FOR_EACH_CELL for (i=1 ; i<_size-1 ; i++) { for (j=1 ; j<_size-1 ; j++) {
#define END_FOR }}	


using namespace std;
using namespace cv;


FluidSimulation::FluidSimulation(int size, float diffusion, float viscosity, int precision, float dt)
{
	_size = size;
	_dt = dt;
	_diff = diffusion;
	_visc = viscosity;
	_precision = precision;
	
	// Dye density (fluid density is constant because incompressible)
	_density = new float[_size * _size];
	_density0 = new float[_size * _size];
	_densityMat = Mat(_size, _size, CV_32FC1, _density);

	// Velocity arrays
	_Vx = new float[_size * _size];
	_Vy = new float[_size * _size];

	// Scratch space for each array so that we can keep old values around while we compute the new ones
	_Vx0 = new float[_size * _size];
	_Vy0 = new float[_size * _size];
	
	Initialize();

}

FluidSimulation::~FluidSimulation()
{
	delete[] _density0;
	delete[] _density;

	delete[] _Vx;
	delete[] _Vy;

	delete[] _Vx0;
	delete[] _Vy0;
	
}

void FluidSimulation::Initialize()
{
		for(int i=0;i<_size * _size;i++)
		{
			_density[i]=0;
			_Vx[i]=0;
			_Vy[i]=0;
		}
}


void FluidSimulation::Initialize(const Mat& d)
{
	if(d.cols == d.rows && d.cols == _size)
	{
		for(int i=0;i<_size * _size;i++)
			_density[i] = static_cast<float>(d.data[i]);
	}
}

void FluidSimulation::Initialize(const Mat& d, const Mat& v)
{
	
	if(d.size()==v.size() && v.channels()==2 && d.cols == _size)
	{
		Mat channel[2];
		split(v, channel);
		
		for(int i=0;i<_size * _size;i++)
		{
			_density[i] = static_cast<float>(d.data[i]);
			_Vx[i] = static_cast<float>(channel[0].data[i]);
			_Vy[i] = static_cast<float>(channel[1].data[i]);
		}
	}
	else
	{
		Initialize(d);
	}

}

void FluidSimulation::setBounds(int boundsFlag, float* x)
{
	int i;

	//free slip boundary edges
	for ( i=1 ; i<_size-1; i++ ) {
		//reverse velocity component on vertical walls (velocX)
		x[coord2idx(0  ,i)]  = (boundsFlag==1) ? -x[coord2idx(1,i)] : x[coord2idx(1,i)];
		x[coord2idx(_size-1,i)] = (boundsFlag==1) ? -x[coord2idx(_size,i)] : x[coord2idx(_size,i)];
		
		//reverse velocity component on horizontal (top and bottom) walls (velocY)
		x[coord2idx(i,0  )]  = (boundsFlag==2) ? -x[coord2idx(i,1)] : x[coord2idx(i,1)];
		x[coord2idx(i,_size-1)] = (boundsFlag==2) ? -x[coord2idx(i,_size)] : x[coord2idx(i,_size)];
	}
	
	//corner conditions
	x[coord2idx(0,      0     )] = 0.5f * (x[coord2idx(1,  0   )] + x[coord2idx(0,      1 )]);
	x[coord2idx(0,      _size - 1)] = 0.5f * (x[coord2idx(1,  _size-1)] + x[coord2idx(0,      _size)]);
	x[coord2idx(_size - 1, 0     )] = 0.5f * (x[coord2idx(_size, 0   )] + x[coord2idx(_size - 1, 1 )]);
	x[coord2idx(_size - 1, _size - 1)] = 0.5f * (x[coord2idx(_size, _size-1)] + x[coord2idx(_size - 1, _size)]);
}


void FluidSimulation::linearSolve( int boundsFlag, float* x, float* x0, float a, float c)
{
	int i, j, k;

	// Gauss-Sidel
	for ( k=0 ; k<_precision ; k++ ) {
		
		FOR_EACH_CELL
			//exchange values with neighbors
			x[coord2idx(i,j)] = (x0[coord2idx(i,j)] + a*(x[coord2idx(i-1,j)] + x[coord2idx(i+1,j)] + x[coord2idx(i,j-1)] + x[coord2idx(i,j+1)])) / c;
		END_FOR
		
		// factor in boundary conditions with each solution iteration
		setBounds(boundsFlag, x); 
	}
}


void FluidSimulation::diffuse(int boundsFlag, float *x, float *x0, float coef)
{
	float diffusionPerCell = _dt * coef * _size * _size;
	linearSolve ( boundsFlag, x, x0, diffusionPerCell, 1+4*diffusionPerCell);
}


void FluidSimulation::project(float *velocX, float *velocY, float *p, float *div)
{
		int i, j;
		float h = 1.0 / _size;
		
		FOR_EACH_CELL
				//calculate initial solution to gradient field based on the difference in velocities of surrounding cells.
				div[coord2idx(i, j)] = -0.5f*h*(velocX[coord2idx(i + 1, j)]	- velocX[coord2idx(i - 1, j)] + velocY[coord2idx(i, j + 1)] - velocY[coord2idx(i, j - 1)]);
				//set projected solution values to be zero
				p[coord2idx(i, j)] = 0;
		END_FOR
	
	//set bounds for diffusion
	setBounds(0, div);
	setBounds(0, p);
	
	// calculate gradient (height) field
	linearSolve(0, p, div, 1, 4);


		FOR_EACH_CELL
		//subtract gradient field from current velocities
				velocX[coord2idx(i, j)] -= 0.5f * (p[coord2idx(i + 1, j)] - p[coord2idx(i - 1, j)]) * _size;
				velocY[coord2idx(i, j)] -= 0.5f * (p[coord2idx(i, j + 1)] - p[coord2idx(i, j - 1)]) * _size;
		END_FOR
		
	//set boundaries for velocity
	setBounds(1, velocX);
	setBounds(2, velocY);	
	
	
}

void FluidSimulation::advect (int boundsFlag, float* d, float* d0, float* velocX, float* velocY)
{
	int i, j;
	int i0; //new x cell coordinate based on velocity grid
	int j0; //new y cell coordinate based on velocity grid
	int i1; //x + 1 cell next to new cell coordinate
	int j1; //y + 1 cell next to new cell coordinate
	float x, y, s0, t0, s1, t1, dt0;

	//initial time differential = dt * number of cells in a row
	dt0 = _dt * _size;

	//back trace density and velocity values from the center of each cell
	FOR_EACH_CELL

		// calculate new coordinates based on existing velocity grids
		x = i - dt0 * velocX[coord2idx(i,j)]; 
		y = j - dt0 * velocY[coord2idx(i,j)];
		
		//limit x coordinate to fall within the grid 
		if (x < 0.5f)      x = 0.5f; 
		if (x > _size -1 + 0.5f) x = _size -1 + 0.5f; 
		i0 = (int)x; 
		i1 = i0 + 1;
		
		//limit y coordinate to fall within the grid
		if (y < 0.5f)      y = 0.5f; 
		if (y > _size -1 + 0.5f) y = _size -1 + 0.5f; 
		j0 = (int)y; 
		j1 = j0 + 1;

		s1 = x - i0; //difference between calculated x position and limited x position
		s0 = 1 - s1; //calculate relative x distance from center of this cell

		t1 = y - j0;  
		t0 = 1 - t1;
		
		//blend several values from the velocity grid together, based on where in the cell 
		//the new coordinate valls
		d[coord2idx(i,j)] = s0 * (t0 * d0[coord2idx(i0,j0)] + t1 * d0[coord2idx(i0,j1)]) +
							s1 * (t0 * d0[coord2idx(i1,j0)] + t1 * d0[coord2idx(i1,j1)]);

	END_FOR
	
	setBounds(boundsFlag, d);
}

void FluidSimulation::AddDensity(int x, int y, float amount)
{
	_density[coord2idx(x, y)] += amount;
}

void FluidSimulation::AddVelocity(int x, int y, float amountX, float amountY)
{
	int index = coord2idx(x, y);

	_Vx[index] += amountX;
	_Vy[index] += amountY;
}

void FluidSimulation::Update()
{
	//*
	// Diffuse density on thread
	std::thread updateDensityThread([&] {diffuse(0, _density0, _density, _diff);});
	
	// Diffuse velocity; Vx on thread, Vy on main
	std::thread diffuseVelocityXThread([&] {diffuse(1, _Vx0, _Vx, _visc);});
	diffuse(2, _Vy0, _Vy, _visc);
	diffuseVelocityXThread.join();
	project(_Vx0, _Vy0, _Vx, _Vy);

	// Advect velocity; Vx on thread, Vy on main
	std::thread advectVelocityXThread([&] {advect(1, _Vx, _Vx0, _Vx0, _Vy0);});
	advect(2, _Vy, _Vy0, _Vx0, _Vy0);
	advectVelocityXThread.join();
	project(_Vx, _Vy, _Vx0, _Vy0);
	
	// Advect density on main
	updateDensityThread.join();
	advect(0, _density, _density0, _Vx, _Vy);
//*/

/*
	// Compute velocity step
	diffuse(1, _Vx0, _Vx, _visc);
	diffuse(2, _Vy0, _Vy, _visc);
	project(_Vx0, _Vy0, _Vx, _Vy);
	
	advect(1, _Vx, _Vx0, _Vx0, _Vy0);
	advect(2, _Vy, _Vy0, _Vx0, _Vy0);
	project(_Vx, _Vy, _Vx0, _Vy0);

	// Compute density step
	diffuse(0, _density0, _density, _diff);
	advect(0, _density, _density0, _Vx, _Vy);
//*/

}

char FluidSimulation::Display(string name ,int fps)
{
	imshow( name, _densityMat );
	char key = waitKey(1000/fps);
	return key;

}

Mat& FluidSimulation::getCurrentState()
{
	return _densityMat;
}

