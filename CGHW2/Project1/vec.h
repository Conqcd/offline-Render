#pragma once

#include<cmath>
#include "common.h"

//#if USEGPU == true
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#endif

class Vec2
{
public:
	float x, y;
public:
	Vec2()
	{
		x = y = 0;
	}
	Vec2(float _x, float _y) :x(_x), y(_y)
	{

	}
	Vec2(float _v[2]) :x(_v[0]), y(_v[1])
	{

	}
	void set(float _x, float _y)
	{
		x = _x; y = _y;
	}
	void set(float _v[2])
	{
		x = _v[0]; y = _v[1];
	}

	Vec2 operator -(const Vec2& v) const
	{
		return Vec2(x -v.x, y -v.y);
	}

	Vec2 operator +(const Vec2& v) const
	{
		return Vec2(x + v.x, y + v.y);
	}
	
	Vec2 operator *(float t) const
	{
		return Vec2(x * t, y * t);
	}
	
	float getx() { return x; }
	float gety() { return y; }
	~Vec2()
	{

	}
};

class Vec3
{
public:
	float x, y, z;
public:
	GPULINE Vec3()
	{
		x = y = z = 0;
	}
	GPULINE Vec3(float _x, float _y, float _z) :x(_x), y(_y), z(_z)
	{

	}
	GPULINE Vec3(float _v[3]) :x(_v[0]), y(_v[1]), z(_v[2])
	{

	}
	GPULINE void set(float _x, float _y, float _z)
	{
		x = _x; y = _y; z = _z;
	}

	GPULINE void set(float _v[3])
	{
		x = _v[0]; y = _v[1]; z = _v[2];
	}
	GPULINE ~Vec3()
	{

	}
	GPULINE float getx() const
	{
		return x;
	}
	GPULINE float gety() const
	{
		return y;
	}
	GPULINE float getz() const
	{
		return z;
	}
	GPULINE Vec3& operator += (const Vec3& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	GPULINE Vec3& operator -= (const Vec3& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	GPULINE Vec3 operator - () const {
		return Vec3(-x, -y, -z);
	}
	GPULINE Vec3 operator+ (const Vec3& v) const
	{
		return Vec3(x + v.x, y + v.y, z + v.z);
	}

	GPULINE float dot(const Vec3& vec) const {
		return x * vec.x + y * vec.y + z * vec.z;
	}
	GPULINE Vec3 ScalarProduct(const Vec3& vec) const {
		return Vec3(x * vec.x, y * vec.y, z * vec.z);
	}
	GPULINE Vec3 cross(const Vec3& vec) const
	{
		return Vec3(y * vec.z - z * vec.y, z * vec.x - x * vec.z, x * vec.y - y * vec.x);
	}

	GPULINE Vec3& operator *= (float t) {
		x *= t;
		y *= t;
		z *= t;
		return *this;
	}
	GPULINE Vec3 operator *(float t) const
	{
		return Vec3(x * t, y * t, z * t);
	}

	GPULINE float operator[](int t) const
	{
		switch (t)
		{
		case 0:
			return x;
			break;
		case 1:
			return y;
			break;
		case 2:
			return z;
			break;
		default:
			break;
		}
	}
	
	GPULINE float length() const {
		return float(sqrt(x * x + y * y + z * z));
	}
	
	GPULINE float square() const {
		return (x * x + y * y + z * z);
	}
	
	GPULINE void unit_vector()
	{
		float sum = x * x + y * y + z * z;
		if (sum > GLH_EPSILON_2) {
			float base = float(1.0 / sqrt(sum));
			x *= base;
			y *= base;
			z *= base;
		}
	}
};

GPULINE inline Vec3 unit_vector(const Vec3& v)
{
	Vec3 vec(v);
	vec.unit_vector();
	return vec;
}

GPULINE inline Vec3 operator * (float t, const Vec3& v) {
	return v * t;
}

GPULINE inline Vec3 operator / (const Vec3& v, float t) {
	return v * (1 / t);
}

GPULINE inline float operator * (const Vec3& v1, const Vec3& v2) {
	return v1.dot(v2);
}

GPULINE inline Vec3 operator - (const Vec3& v1, const Vec3& v2) {
	return v1+(-v2);
}

GPULINE inline void get_sphere_uv(const Vec3& p, double& u, double& v)
{
	auto phi = atan2(p.getz(), p.getx());
	auto theta = asin(p.gety());
	u = 1 - (phi + PI) / (2 * PI);
	v = (theta + PI / 2) / PI;
}

GPULINE inline Vec3 cross(const Vec3& vec1, const Vec3& Vec2)
{
	return vec1.cross(Vec2);
}

GPULINE inline float dot(const Vec3& vec1, const Vec3& Vec2)
{
	return vec1.dot(Vec2);
}

GPULINE inline Vec3 barycentric(const Vec2& v1, const Vec2& v2, const Vec2& v3,const Vec2& point)
{
	Vec3 vec= cross(Vec3(v3.x - v1.x, v2.x - v1.x, v1.x - point.x), Vec3(v3.y - v1.y, v2.y - v1.y, v1.y - point.y));
	if (std::abs(vec.z) < 1) return Vec3(-1, 1, 1);
	return Vec3(1.f - (vec.x + vec.y) / vec.z, vec.y / vec.z, vec.x / vec.z);
}

GPULINE inline Vec3 barycentric(const Vec3& v1, const Vec3& v2, const Vec3& v3, const Vec3& point)
{
	Vec3 vec1 = point - v3;
	Vec3 vec2 = v1 - v3;
	Vec3 vec3 = v2 - v3;
	double area = vec2.cross(vec3).length();
	if(area < GLH_EPSILON_2)
		return Vec3(-1, 1, 1);
	double area1=vec2.cross(vec1).length(),area2=vec1.cross(vec3).length()/2;
	return Vec3(area2 / area, area1 / area, 1 - area2 / area - area1 / area);
}

GPULINE inline void barycentric(const Vec3& v1, const Vec3& v2, const Vec3& v3, const Vec2& point,Vec3& t1, Vec3& t2,Vec3& b)
{
	t1.set(v3.x - v1.x, v2.x - v1.x, v1.x - point.x);
	t2.set(v3.y - v1.y, v2.y - v1.y, v1.y - point.y);
	b.set(t1.y * t2.z - t1.z * t2.y, t1.z * t2.x - t1.x * t2.z, t1.x * t2.y - t1.y * t2.x);
	if (fabs(b.z) < GLH_EPSILON) b.set(-1, 1, 1);
	else b.set(1.f - (b.x + b.y) / b.z, b.y / b.z, b.x / b.z);
}


GPULINE inline Vec3 random_unit_vector()
{
	auto a = random_double(0, 2 * PI);
	auto z = random_double(-1, 1);
	auto r = sqrt(1 - z * z);
	return Vec3(r * cos(a), r * sin(a), z);
}

#if USEGPU==true

__device__ inline Vec3 gpu_random_cosine_direction(curandState* curand_states, int width)
{
	auto r1 = gpu_random_double(curand_states, width);
	auto r2 = gpu_random_double(curand_states, width);
	auto z = sqrtf(1 - r2);

	auto phi = 2 * PI * r1;
	auto x = cosf(phi) * sqrtf(r2);
	auto y = sinf(phi) * sqrtf(r2);

	return Vec3(x, y, z);
}

__device__ inline Vec3 gpu_random_cosine_n_direction(curandState* curand_states, int width,int Ns)
{
	auto r1 = gpu_random_double(curand_states, width);
	auto r2 = gpu_random_double(curand_states, width);

	float cos_phi = cosf(2.0 * PI * r1);
	float sin_phi = sinf(2.0 * PI * r1);
	float z = powf((1.0 - r2), 1.0 / (Ns + 1.0));
	float sin_theta = sqrtf(1.0 - z * z);
	float x = sin_theta * cos_phi;
	float y = sin_theta * sin_phi;
	
	return Vec3(x, y, z);
}
#endif

GPULINE inline Vec3 reflect(const Vec3& v, const Vec3& n)
{
	return v - 2 * dot(v, n) * n;
}


GPULINE inline Vec3 random_cosine_direction()
{
	auto r1 = random_double();
	auto r2 = random_double();
	auto z = sqrt(1 - r2);

	auto phi = 2 * PI * r1;
	auto x = cos(phi) * sqrt(r2);
	auto y = sin(phi) * sqrt(r2);

	return Vec3(x, y, z);
}

GPULINE inline Vec3 random_to_sphere(float r, float sq)
{
	auto r1 = random_double();
	auto r2 = random_double();
	auto z = 1 + r2 * (sqrt(1 - r * r / sq) - 1);

	auto phi = 2 * PI * r1;
	auto x = cos(phi) * sqrt(1 - z * z);
	auto y = sin(phi) * sqrt(1 - z * z);

	return Vec3(x, y, z);
}