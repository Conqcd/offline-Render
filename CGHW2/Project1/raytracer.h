#pragma once

#include <memory>

#include "image.h"
#include "model.h"
using namespace std;

#if VERSION==0

extern void RayTracing(const GPU_DataSet& model, int depth, SDLImage& screen, int samples_per_pixel,
	DoubleColor background);
extern GPU_DataSet CreateGPUModel(const Model& model,int height,int width);

class Tracer
{
public:
	SDLImage screen;
	int width, height;
	int samples_per_pixel;
	DoubleColor background={0.0,0,0};
	bool stop = false;
#if USEGPU==false
	bool useGPU = false;
#else
	bool useGPU = true;

#endif

public:
	Tracer();
	Tracer(int _width, int _height, int spp);
	void tracing(const Model& model, int depth, const char* path_result);
	DoubleColor ray_color(const Ray& r, const DoubleColor& background, const Model& model, int depth);
	void show();
	~Tracer()
	{
		
	}
};

#endif