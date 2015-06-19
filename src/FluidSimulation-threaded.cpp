#include "FluidSimulation.hpp"
#include <cstdlib>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>//imshow

#define NBTHREAD 3


using namespace std;
using namespace cv;


FluidSimulation::FluidSimulation(int size, float diffusion, float viscosity, int precision, float dt)
{
	_size = size;
	_dt = dt;
	_diff = diffusion;
	_visc = viscosity;
	_simulationPrecision = precision;

	_s = new float[_size * _size];
	
	// Dye density (fluid density is constant because incompressible)
	_density = new float[_size * _size];

	// Velocity arrays
	_Vx = new float[_size * _size];
	_Vy = new float[_size * _size];

	// Scratch space for each array so that we can keep old values around while we compute the new ones
	_Vx0 = new float[_size * _size];
	_Vy0 = new float[_size * _size];
	
	_densityMat = cv::Mat(_size, _size, CV_32FC1, _density);
	
	Initialize();

}

FluidSimulation::~FluidSimulation()
{
	delete[] _s;
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


void FluidSimulation::set_bnd(int b, float *x)
{

		for (int i = 1; i < _size - 1; i++) {
			x[coord2idx(i, 0)] = (b == 2) ? -x[coord2idx(i, 1)] : x[coord2idx(i, 1)];
			x[coord2idx(i, _size - 1)] = (b == 2) ? -x[coord2idx(i, _size - 2)] : x[coord2idx(i, _size - 2)];
		}

		for (int j = 1; j < _size - 1; j++) {
			x[coord2idx(0, j)] = (b == 1) ? -x[coord2idx(1, j)] : x[coord2idx(1, j)];
			x[coord2idx(_size - 1, j)] = (b == 1) ? -x[coord2idx(_size - 2, j)] : x[coord2idx(_size - 2, j)];
		}
	
	//take average on corners
	x[coord2idx(0, 0)] = 0.5f * ( x[coord2idx(1, 0)] + x[coord2idx(0, 1)] );
	x[coord2idx(0, _size - 1)] = 0.5f * (x[coord2idx(1, _size - 1)] + x[coord2idx(0, _size - 2)] );
	x[coord2idx(_size - 1, 0)] = 0.5f * (x[coord2idx(_size - 2, 0)] + x[coord2idx(_size - 1, 1)] );		
	x[coord2idx(_size - 1, _size - 1)] = 0.5f * ( x[coord2idx(_size - 1, _size - 2)] + x[coord2idx(_size - 2, _size - 1)]);
}


void FluidSimulation::lin_solve(int b, float *x, float *x0, float a, float c)
{

      std::thread *tt = new std::thread[NBTHREAD];
      
         //Lauch parts-1 threads
         for (int i = 0; i < NBTHREAD; ++i) {
             tt[i] = std::thread(tst, &image, &image2, bnd[i], bnd[i + 1]);
         }
 
        //Use the main thread to do part of the work !!!
        tst(&image, &image2, bnd[i], bnd[i + 1]);
             
         //Join parts-1 threads
         for (int i = 0; i < NBTHREAD; ++i)
             tt[i].join();

	
	
	
	c = 1.0 / c;
	
	for (int k = 0; k < _simulationPrecision; k++) {
		
			for (int j = 1; j < _size - 1; j++) {
				for (int i = 1; i < _size - 1; i++) {
					x[coord2idx(i, j)] =
						(x0[coord2idx(i, j)]
						+ a*(x[coord2idx(i + 1, j)]
						+ x[coord2idx(i - 1, j)]
						+ x[coord2idx(i, j + 1)]
						+ x[coord2idx(i, j - 1)]
						)) * c;
				}
			}
			
		}
		
		set_bnd(b, x);
	
}


void FluidSimulation::diffuse(int b, float *x, float *x0, float diff)
{
	float a = _dt * diff * (_size - 2) * (_size - 2);
	lin_solve(b, x, x0, a, 1 + 4 * a);
}


void FluidSimulation::project(float *velocX, float *velocY, float *p, float *div)
{

		for (int j = 1; j < _size - 1; j++) {
			for (int i = 1; i < _size - 1; i++) {
				div[coord2idx(i, j)] = -0.5f*(
					velocX[coord2idx(i + 1, j)]
					- velocX[coord2idx(i - 1, j)]
					+ velocY[coord2idx(i, j + 1)]
					- velocY[coord2idx(i, j - 1)]
					) / _size;
				p[coord2idx(i, j)] = 0;
			}
		}
	
	set_bnd(0, div);
	set_bnd(0, p);
	lin_solve(0, p, div, 1, 4);


		for (int j = 1; j < _size - 1; j++) {
			for (int i = 1; i < _size - 1; i++) {
				velocX[coord2idx(i, j)] -= 0.5f * (p[coord2idx(i + 1, j)] - p[coord2idx(i - 1, j)]) * _size;
				velocY[coord2idx(i, j)] -= 0.5f * (p[coord2idx(i, j + 1)] - p[coord2idx(i, j - 1)]) * _size;
			}
		}
	
	set_bnd(1, velocX);
	set_bnd(2, velocY);
}


void FluidSimulation::advect(int b, float *d, float *d0, float *velocX, float *velocY)
{
	float i0, i1, j0, j1;

	float dtx = _dt * (_size - 2);
	float dty = _dt * (_size - 2);
	
	float s0, s1, t0, t1;
	float x, y;

	float Nfloat = _size;
	float ifloat, jfloat;
	int i, j;

		for (j = 1, jfloat = 1; j < _size - 1; j++, jfloat++) {
			for (i = 1, ifloat = 1; i < _size - 1; i++, ifloat++) {
				
				x = ifloat - dtx * velocX[coord2idx(i, j)];
				y = jfloat - dty * velocY[coord2idx(i, j)];

				if (x < 0.5f) x = 0.5f;
				if (x > Nfloat + 0.5f) x = Nfloat + 0.5f;								
				i0 = floorf(x);
				i1 = i0 + 1.0f;
				
				if (y < 0.5f) y = 0.5f;
				if (y > Nfloat + 0.5f) y = Nfloat + 0.5f;
				j0 = floorf(y);
				j1 = j0 + 1.0f;

				s0 = 1.0f - x + floorf(x);
				t0 = 1.0f - y + floorf(y);
				s1 = x - i0;
				t1 = y - j0;
				
				
				int i0i = i0;
				int i1i = i1;
				int j0i = j0;
				int j1i = j1;

				d[coord2idx(i, j)] =

					s0 * (t0 * d0[coord2idx(i0i, j0i)] 
						+ t1 * d0[coord2idx(i0i, j1i)] )
						
					+s1 * (t0 * d0[coord2idx(i1i, j0i)] 
						+ t1 * d0[coord2idx(i1i, j1i)] );
			}
		}
		
	set_bnd(b, d);
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

void FluidSimulation::Step()
{
	// Compute velocity step
	diffuse(1, _Vx0, _Vx, _visc);
	diffuse(2, _Vy0, _Vy, _visc);
	project(_Vx0, _Vy0, _Vx, _Vy);

	advect(1, _Vx, _Vx0, _Vx0, _Vy0);
	advect(2, _Vy, _Vy0, _Vx0, _Vy0);
	project(_Vx, _Vy, _Vx0, _Vy0);

	// Compute density step
	diffuse(0, _s, _density, _diff);
	advect(0, _density, _s, _Vx, _Vy);
}

char FluidSimulation::Display(string name, int fps)
{
	imshow( name, _densityMat );
	char key = waitKey(1000/fps);
	return key;
}

Mat& FluidSimulation::getCurrentState()
{
	return _densityMat;
}

