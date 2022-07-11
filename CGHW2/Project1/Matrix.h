#pragma once
#include"vec.h"
#include <algorithm>
using namespace std;

class mat4
{
public:
	float _data[4][4];
public:
	mat4()
	{
		zero();
	}
	mat4(float num)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if (i == j)
					_data[i][j] = num;
				else
					_data[i][j] = 0;
			}
		}
	}
	~mat4()
	{

	}
	void zero()
	{
		memset(&_data[0][0], 0, sizeof(float) * 16);
	}

	float* operator[] (int x)
	{
		return _data[x];
	}
	const float* operator[] (int x) const
	{
		return _data[x];
	}
	mat4 operator*(const mat4& rhs) const{
		mat4 result;
		for (int r = 0; r < 4; ++r) {
			for (int c = 0; c < 4; ++c) {
				float val = 0;
				for (int i = 0; i < 4; ++i) {
					val += operator[](r)[i] * rhs[i][c];
				}
				result[r][c] = val;
			}
		}
		return result;
	}
	void dotV3(const Vec3& src,Vec3& dst)const{

		float v = _data[3][0] * src.x + _data[3][1] * src.y + _data[3][2] * src.z + _data[3][3];
		if(fabs(v)> GLH_EPSILON)
		{
			dst.x = (_data[0][0] * src.x + _data[0][1] * src.y + _data[0][2] * src.z + _data[0][3]) / v;
			dst.y = (_data[1][0] * src.x + _data[1][1] * src.y + _data[1][2] * src.z + _data[1][3]) / v;
			dst.z = (_data[2][0] * src.x + _data[2][1] * src.y + _data[2][2] * src.z + _data[2][3]) / v;
		}else
		{
			dst.x = (_data[0][0] * src.x + _data[0][1] * src.y + _data[0][2] * src.z + _data[0][3]) / GLH_EPSILON;
			dst.y = (_data[1][0] * src.x + _data[1][1] * src.y + _data[1][2] * src.z + _data[1][3]) / GLH_EPSILON;
			dst.z = (_data[2][0] * src.x + _data[2][1] * src.y + _data[2][2] * src.z + _data[2][3]) / GLH_EPSILON;
		}
	}
	void dotV3_Special(const Vec3& src, Vec3& dst)const {

		float v = _data[3][0] * src.x + _data[3][1] * src.y + _data[3][2] * src.z + _data[3][3];
		if (fabs(v) > GLH_EPSILON)
		{
			dst.x = (_data[0][0] * src.x + _data[0][1] * src.y + _data[0][2] * src.z + _data[0][3]) / v;
			dst.y = (_data[1][0] * src.x + _data[1][1] * src.y + _data[1][2] * src.z + _data[1][3]) / v;
			dst.z = (_data[2][0] * src.x + _data[2][1] * src.y + _data[2][2] * src.z + _data[2][3]);
		}
		else
		{
			dst.x = 0;
			dst.y = 0;
			dst.z = 0;
		}
	}
};

inline mat4 viewport(int x, int y, int w, int h) {
	mat4 view_port_matrix (1.0f);
	view_port_matrix[0][3] = -x;
	view_port_matrix[1][3] = -y;
	view_port_matrix[0][0] = w / 2.;
	view_port_matrix[1][1] = h / 2.;
	return view_port_matrix;
}

inline mat4 perspective(float Zoom, float aspect, float zNear, float zFar)
{
	const float tangent = tan(Zoom / 2);

	mat4 Result;
	float height = zNear * tangent;
	float width = height * aspect;

	double l = -width, r = width, b = -height, t = height;

	Result[0][0] = 2 * zNear / (r - l);
	Result[0][2] = (r + l) / (r - l);
	Result[1][1] = 2 * zNear / (t - b);
	Result[1][2] = (t + b) / (t - b);
	/*Result[2][2] = -2 / (zFar - zNear);
	Result[2][3] = (zFar + zNear) / (zFar - zNear);*/
	Result[2][2] = (zFar + zNear) / (zFar - zNear);
	Result[2][3] = -2 * zFar * zNear / (zFar - zNear);
	Result[3][2] = -1;
	
	//Result[0][0] = 1 / tangent / aspect;
	//Result[1][1] = 1 / tangent;
	//Result[2][2] = -(zFar + zNear) / (zFar - zNear);
	//Result[2][3] = -(2.0f * zFar * zNear) / (zFar - zNear);
	//Result[3][2] = -1.0f;
	//Result[3][3] = 0.0f;

	return Result;
}

inline mat4 translate(const mat4& m, const Vec3& vec)
{
	mat4 m2 = m;
	m2[0][3] = vec.getx() * m[0][0] + vec.gety() * m[0][1] + vec.getz() * m[0][2];
	m2[1][3] = vec.getx() * m[1][0] + vec.gety() * m[1][1] + vec.getz() * m[1][2];
	m2[2][3] = vec.getx() * m[2][0] + vec.gety() * m[2][1] + vec.getz() * m[2][2];
	return m2;
}

inline mat4 rotate(const mat4& m, float angle, const Vec3& vec)
{
	const float a = angle;
	const float c = cos(a);
	const float s = sin(a);

	Vec3 axis(unit_vector(vec));
	Vec3 temp((1.0 - c) * axis);

	mat4 Rotate(1.0f);
	Rotate[0][0] = c + temp[0] * axis[0];
	Rotate[0][1] = temp[0] * axis[1] + s * axis[2];
	Rotate[0][2] = temp[0] * axis[2] - s * axis[1];

	Rotate[1][0] = temp[1] * axis[0] - s * axis[2];
	Rotate[1][1] = c + temp[1] * axis[1];
	Rotate[1][2] = temp[1] * axis[2] + s * axis[0];

	Rotate[2][0] = temp[2] * axis[0] + s * axis[1];
	Rotate[2][1] = temp[2] * axis[1] - s * axis[0];
	Rotate[2][2] = c + temp[2] * axis[2];

	mat4 Result(1.0f);
	Result[0][0] = m[0][0] * Rotate[0][0] + m[0][1] * Rotate[0][1] + m[0][2] * Rotate[0][2];
	Result[1][0] = m[1][0] * Rotate[0][0] + m[1][1] * Rotate[0][1] + m[1][2] * Rotate[0][2];
	Result[2][0] = m[2][0] * Rotate[0][0] + m[2][1] * Rotate[0][1] + m[2][2] * Rotate[0][2];
	Result[0][1] = m[0][0] * Rotate[1][0] + m[0][1] * Rotate[1][1] + m[0][2] * Rotate[1][2];
	Result[1][1] = m[1][0] * Rotate[1][0] + m[1][1] * Rotate[1][1] + m[1][2] * Rotate[1][2];
	Result[2][1] = m[2][0] * Rotate[1][0] + m[2][1] * Rotate[1][1] + m[2][2] * Rotate[1][2];
	Result[0][2] = m[0][0] * Rotate[2][0] + m[0][1] * Rotate[2][1] + m[0][2] * Rotate[2][2];
	Result[1][2] = m[1][0] * Rotate[2][0] + m[1][1] * Rotate[2][1] + m[1][2] * Rotate[2][2];
	Result[2][2] = m[2][0] * Rotate[2][0] + m[2][1] * Rotate[2][1] + m[2][2] * Rotate[2][2];
	Result[0][3] = m[0][3];
	Result[1][3] = m[1][3];
	Result[2][3] = m[2][3];
	return Result;
}

inline mat4 scale(const mat4& m, const Vec3& vec)
{
	mat4 m2 = m;
	m2[0][0] *= vec.getx();
	m2[1][0] *= vec.getx();
	m2[2][0] *= vec.getx();
	m2[0][1] *= vec.gety();
	m2[1][1] *= vec.gety();
	m2[2][1] *= vec.gety();
	m2[0][2] *= vec.getz();
	m2[1][2] *= vec.getz();
	m2[2][2] *= vec.getz();
	return m2;
}