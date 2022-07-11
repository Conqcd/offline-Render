#pragma once
#include "common.h"
#include "vec.h"

class Ray
{
public:
	Vec3 origin, direction;
	double time;
public:
	GPULINE Ray(){}
	GPULINE Ray(const Vec3& ori, const Vec3& dir, double tm = 0):origin(ori),direction(dir),time(tm)
	{
		
	}
	GPULINE Vec3 GetOrigin() const
	{
		return origin;
	}
	GPULINE Vec3 GetDirection() const
	{
		return direction;
	}
	GPULINE double GetTime() const
	{
		return time;
	}

	GPULINE Vec3 PointAt(double t) const
	{
		return origin+t*direction;
	}
};