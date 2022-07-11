#pragma once
#include "onb.h"

#if USEGPU==true

extern struct hit_record;
extern class GPU_DataSet;

class PDF
{
public:
	GPULINE PDF()
	{
	}
	GPULINE PDF(int type) :type(type)
	{
	}
	GPULINE PDF(const Vec3& w,int ns):type(0),Ns(ns)
	{
		uvw.build_from_w(w);
	}
	GPULINE PDF(GPU_DataSet* pt, const Vec3& p) :ptr(pt), origin(p), type(1)
	{
		
	}
	GPULINE PDF(PDF* p0, PDF* p1,float w) : type(2), weight(w){
		pptr[0] = p0;
		pptr[1] = p1;
	}
	__device__ double value(const Vec3& direction)const;
	__device__ double value(const Ray& r, float infinity,bool& hit,hit_record& needTraverse)const;

	__device__ Vec3 generate(curandState* curand_states, int width) const;
	__device__ Vec3 generate(curandState* curand_states, int width, const Vec3& o) const;

	GPULINE ~PDF() {}
public:
	ONB uvw;
	int type;
	GPU_DataSet* ptr;
	Vec3 origin;
	PDF* pptr[2];
	float weight;
	int Ns;
	/*
	 * type是pdf的类型
	 * 0:cosine_pdf
	 * 1:hittable_pdf
	 * 2:mixture_pdf
	 */
};


#else


class PDF
{
public:
	GPULINE virtual double value(const Vec3& direction)const = 0;


	GPULINE virtual Vec3 generate() const = 0;

	GPULINE virtual ~PDF() {}
};

class Cosine_PDF :public PDF
{
public:
	GPULINE Cosine_PDF(const Vec3& w) { uvw.build_from_w(w); }

	GPULINE virtual double value(const Vec3& direction)const override
	{
		auto cosine = dot(uvw.w(), direction);
		return cosine < 0 ? 0 : cosine / PI;
	}

	GPULINE virtual Vec3 generate() const override
	{
		return uvw.local(random_cosine_direction());
	}

public:
	ONB uvw;
};

#endif
