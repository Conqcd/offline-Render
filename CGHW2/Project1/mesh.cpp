#include "mesh.h"

Mesh::Mesh()
{

}

Mesh::Mesh(unsigned int _numvtx, unsigned int _numtri, unsigned int _numind,
	Triangle* _face,
	Vec3* _vertex,
	Vec3* _normal,
	Vec2* _texcord)
{
	numvtx = _numvtx;
	numind = _numind;
	numtri = _numtri;
	face = _face;
	vertex = _vertex;
	normal = _normal;
	texcord = _texcord;
}

Mesh::~Mesh()
{
	delete[] face;
	delete[] vertex;
	delete[] normal;
	delete[] texcord;
}

void Mesh::draw(Shader* shader)
{
	/*for (int i = 0; i < numtri; i++) {
			shader->drawTriangle(vertex[face[i].vid[0]], vertex[face[i].vid[1]], vertex[face[i].vid[2]],
				Color(rand() % 255, rand() % 255, rand() % 255),true);
	}*/
}
