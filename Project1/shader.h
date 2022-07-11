#pragma once

#include "image.h"
#include "box.h"
#include "hierarchyZbuffer.h"
#include <string>
using namespace std;


class Shader
{
public:
	SDLImage screen;
	HierarchyZbuffer *HZB;
	int width, height;
	bool useHiZbuffer;
public:
	Shader();
	Shader(int _width, int _height, int bpp);
	bool drawTriangle(const Vec3& t0, const Vec3& t1, const Vec3& t2,const AABB3& box, Color color, bool changed);
	void show();
	~Shader();
	void Switch();
};