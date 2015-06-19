#ifndef __fluidSimulation__
#define __fluidSimulation__

#include <opencv/cv.hpp>//namespace

class FluidSimulation
{
	private:
	
	int _size;
	int _precision;
    float _dt;
    float _diff;
    float _visc;
    
    cv::Mat _densityMat;
    
    float *_density;
    float *_density0;
    
    float *_Vx;
    float *_Vy;

    float *_Vx0;
    float *_Vy0;
    
    void setBounds(int b, float *x);
	void linearSolve(int b, float *x, float *x0, float a, float c);
	void diffuse(int b, float *x, float *x0, float diff);
	void project(float *velocX, float *velocY, float *p, float *div);
	void advect(int b, float *d, float *d0, float *velocX, float *velocY);
	
	inline int coord2idx( int x, int y) 
	{
		return x + y*_size;
	}
	
	public:
	
	FluidSimulation(int size, float diffusion, float viscosity, int precision, float dt);
	~FluidSimulation();
	
	void AddDensity(int x, int y, float amount);
	void AddVelocity(int x, int y, float amountX, float amountY);
	void Update();
	void Initialize();
	void Initialize(const cv::Mat& d);
	void Initialize(const cv::Mat& d, const cv::Mat& v);
	char Display(std::string name="fluid",int fps=60);
	cv::Mat& getCurrentState();
	
};



#endif
