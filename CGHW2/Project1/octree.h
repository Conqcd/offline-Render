#pragma once
#include "mesh.h"
#include "shader.h"
#include <vector>
using namespace std;

#define Divnum 5

struct OCTnode
{
	AABB3 box;
	OCTnode* _child[8];
	OCTnode* parent;
	vector<unsigned int> face;
	bool isleaf;
	int deep;
	int count;
	OCTnode(const AABB3& _box):
		box(_box)
	{
		isleaf = true;
		for(int i=0;i<8;i++)
		{
			_child[i]=nullptr;
		}
		parent = nullptr;
		deep = count = 0;
	}
	void CreateLeves();
	bool InsertAABB(vector<AABB3>& boxes, vector<OCTnode*>& nodes, unsigned int index);
	void Draw(vector<bool>& culledface, vector<bool>& culledface2, vector<Vec3>& nowvertex, vector<Triangle>& faces, Shader* shader, vector<float>& newz, vector<AABB3>& boxes, bool changed);
	void Draw2(vector<bool>& culledface, vector<bool>& culledface2, vector<Vec3>& nowvertex, vector<Triangle>& faces, Shader* shader, vector<float>& newz, vector<AABB3>& boxes, bool changed);
	void deleteTri(int index);
	void deleteLeave();
	void UpdateTri(vector<AABB3>& boxes, vector<OCTnode*>& nodes, bool parent, unsigned int index);
};

class OCTree
{
public:
	OCTnode* root;
public:
	OCTree()
	{
		root = nullptr;
	}
	OCTree(const AABB3& box)
	{
		createTree(box);
	}
	~OCTree();
	void createTree(const AABB3& box);
	void PostdeleteNode(OCTnode* root);
	void Update(vector<AABB3>& boxes, vector<OCTnode*>& nodes, int numtri,bool first);
private:
	void PostDelete(OCTnode* root);
};