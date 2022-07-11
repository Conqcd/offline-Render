#pragma once

#include"triangle.h"
#include "shader.h"
#include "box.h"

extern struct OCTnode;
extern class OCTree;

class Mesh
{
private:
	Vec3* normal;
	Vec2* texcord;
public:
	Triangle* face;
	Vec3* vertex;
	unsigned int numvtx, numtri, numind;
	Mesh();
	Mesh(unsigned int _numvtx, unsigned int _numtri, unsigned int _numind,
		Triangle* _face,
		Vec3* _vertex,
		Vec3* _normal,
		Vec2* _texcord);
	~Mesh();
	void draw(Shader* shader);
};