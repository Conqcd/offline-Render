#pragma once
#include "ray.h"
#include "common.h"

class ONB
{

public:
	GPULINE ONB() {}

	GPULINE inline Vec3 operator[](int i) const {
		return axis[i];
	}

	GPULINE Vec3 u()const {
		return axis[0];
	}

	GPULINE Vec3 v()const {
		return axis[1];
	}

	GPULINE Vec3 w()const {
		return axis[2];
	}

	GPULINE Vec3 local(double x, double y, double z)const
	{
		return axis[0] * x + axis[1] * y + axis[2] * z;
	}

	GPULINE Vec3 local(const Vec3& a)const
	{
		return axis[0] * a.getx() + axis[1] * a.gety() + axis[2] * a.getz();
	}

	GPULINE void build_from_w(const Vec3& n)
	{
		axis[2] = unit_vector(n);
		Vec3 a = (fabs(w().getx()) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
		axis[1] = unit_vector(cross(w(), a));
		axis[0] = cross(w(), v());
	}


public:
	Vec3 axis[3];
};