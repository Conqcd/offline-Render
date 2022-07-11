#pragma once
#include<cmath>
#include <random>

#define  PI 3.1415926

#define     GLH_EPSILON          float(10e-6)
#define		GLH_EPSILON_2		float(10e-12)

#define     EPSILON          float(10e-3)

#define CAMERA 3
#define OBJECT 2
#define VERSION 0
#define USEGPU true


#if USEGPU==false

#define GPULINE

#else

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "cuda_texture_types.h"
#define __CUDACC__
#include "texture_indirect_functions.h"
#include <cuda.h>
#include <common/cpu_anim.h>


#define GPULINE __host__ __device__

__device__ inline double gpu_random_double(curandState* curand_states, int width)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    return curand_uniform_double(curand_states + idx + idy * width);
}

__device__ inline double gpu_random_double_range(curandState* curand_states, int width,double x1,double x2)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	return curand_uniform_double(curand_states + idx + idy * width) * (x2 - x1) + x1;
}

#endif

const double infinity = std::numeric_limits<double>::infinity();

GPULINE inline double radians(double angle)
{
	return angle * PI / 180.0;
}

inline double random_double()
{
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

inline double random_double(double min, double max)
{
	return random_double() * (max - min) + min;
}


inline void clamp(float& v, double min, double max)
{
	if (v < min)
		v = min;
	else if (v > max)
		v = max;
}


GPULINE inline void swap_float(float& t1, float& t2)
{
	float temp = t1;
	t1 = t2;
	t2 = temp;
}