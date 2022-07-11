#include "octree.h"

void OCTnode::CreateLeves()
{
	deep++;
	float midx = (box._min.x + box._max.x) / 2, midy = (box._min.y + box._max.y) / 2, midz = (box._min.z + box._max.z) / 2;
	_child[0] = new OCTnode(AABB3(Vec3(box._min.x, midy, midz), Vec3(midx, box._max.y, box._max.z)));
	_child[1] = new OCTnode(AABB3(Vec3(midx, midy, midz), Vec3(box._max.x, box._max.y, box._max.z)));
	_child[2] = new OCTnode(AABB3(Vec3(box._min.x, box._min.y, midz), Vec3(midx, midy, box._max.z)));
	_child[3] = new OCTnode(AABB3(Vec3(midx, box._min.y, midz), Vec3(box._max.x, midy, box._max.z)));
	_child[4] = new OCTnode(AABB3(Vec3(box._min.x, midy, box._min.z), Vec3(midx, box._max.y, midz)));
	_child[5] = new OCTnode(AABB3(Vec3(midx, midy, box._min.z), Vec3(box._max.x, box._max.y, midz)));
	_child[6] = new OCTnode(AABB3(Vec3(box._min.x, box._min.y, box._min.z), Vec3(midx, midy, midz)));
	_child[7] = new OCTnode(AABB3(Vec3(midx, box._min.y, box._min.z), Vec3(box._max.x, midy, midz)));

	_child[0]->parent = this;
	_child[1]->parent = this;
	_child[2]->parent = this;
	_child[3]->parent = this;
	_child[4]->parent = this;
	_child[5]->parent = this;
	_child[6]->parent = this;
	_child[7]->parent = this;
}

bool OCTnode::InsertAABB(vector<AABB3>& boxes, vector<OCTnode*>& nodes, unsigned int index)
{
	if(!box.contain(boxes[index]))
	{
		return false;
	}
	count++;
	if(isleaf)
	{
		face.push_back(index);
		nodes[index] = this;
		if (face.size() == Divnum)
		{
			CreateLeves();
			isleaf = false;
			for (int i = 0; i < face.size(); i++)
			{
				bool flag = false;
				for (int j = 0; j < 8; j++)
				{
					if (_child[j]->InsertAABB(boxes,nodes, face[i]))
					{
						flag = true;
						break;
					}
				}
				if(flag)
				{
					int index = face.back();
					face[i] = index;
					face.pop_back();
					i--;
				}
			}
		}
	}else
	{
		bool flag = true;
		for (int i = 0; i < 8; i++)
		{
			if (_child[i]->InsertAABB(boxes, nodes, index))
			{
				flag = false;
				break;
			}
		}
		if (flag)
		{
			face.push_back(index);
			nodes[index] = this;
		}
	}
	return true;
}

void OCTnode::Draw(vector<bool>& culledface, vector<bool>& culledface2, vector<Vec3>& nowvertex, vector<Triangle>& faces,Shader* shader,vector<float>& newz,vector<AABB3>& boxes ,bool changed)
{
	for (int i = 0; i < face.size(); i++) {
		if (culledface[face[i]])
		{
			culledface2[i] = shader->drawTriangle(nowvertex[faces[face[i]].vid[0]], nowvertex[faces[face[i]].vid[1]], nowvertex[faces[face[i]].vid[2]],
				boxes[face[i]], Color(rand() % 255, rand() % 255, rand() % 255), changed);
		}
	}
	for (int i = 0; i < face.size(); i++) {
		if (!culledface[face[i]])
		{
			culledface2[i] = shader->drawTriangle(nowvertex[faces[face[i]].vid[0]], nowvertex[faces[face[i]].vid[1]], nowvertex[faces[face[i]].vid[2]],
				boxes[face[i]], Color(rand() % 255, rand() % 255, rand() % 255), changed);
		}
	}
	for (int i = 0; i < face.size(); i++) {
		culledface[face[i]] = culledface2[face[i]];
	}
}

void OCTnode::Draw2(vector<bool>& culledface, vector<bool>& culledface2, vector<Vec3>& nowvertex,
	vector<Triangle>& faces, Shader* shader, vector<float>& newz, vector<AABB3>& boxes, bool changed)
{
	for (int i = 0; i < face.size(); i++) {
			culledface[i] = shader->drawTriangle(nowvertex[faces[face[i]].vid[0]], nowvertex[faces[face[i]].vid[1]], nowvertex[faces[face[i]].vid[2]],
				boxes[face[i]], Color(rand() % 255, rand() % 255, rand() % 255), changed);
	}

}

void OCTnode::deleteTri(int index)
{
	for(int i=0;i<face.size();i++)
	{
		if(index==face[i])
		{
			int temp = face.back();
			face[i] = temp;
			face.pop_back();
			break;
		}
	}
}

void OCTnode::deleteLeave()
{
	for (int i = 0; i < 8; i++)
	{
		delete _child[i];
		_child[i] = nullptr;
	}
	isleaf = true;
}

void OCTree::PostdeleteNode(OCTnode* root)
{
	if (root->_child[0] == nullptr)
		return;
	int count = 0;
	for (int i = 0; i < 8; i++)
	{
		PostdeleteNode(root->_child[i]);
		count += root->_child[i]->count;
	}
	if (count == 0)
		root->deleteLeave();
}

void OCTnode::UpdateTri(vector<AABB3>& boxes, vector<OCTnode*>& nodes,bool parents, unsigned index)
{
	if(box.contain(boxes[index]))
	{
		if(!isleaf)
		{
			if(parents)
			{
				face.push_back(index);
				nodes[index] = this;
			}
			int sum = 0;
			int flag = false;
			for (int j = 0; j < 8; j++)
			{
				sum += _child[j]->count;
				if (_child[j]->InsertAABB(boxes, nodes, index))
				{
					flag = true;
					break;
				}
			}
			if (flag)
			{
				deleteTri(index);
			}else
			{
				if(sum==0)
				{
					deleteLeave();
					isleaf = true;
				}
			}
		}
	}
	else
	{
		count--;
		if (!parents)
			deleteTri(index);

		if (parent != nullptr)
			parent->UpdateTri(boxes, nodes, true, index);
		else
			nodes[index] = nullptr;
	}
}

OCTree::~OCTree()
{
	PostDelete(root);
}

void OCTree::createTree(const AABB3& box)
{
	root = new OCTnode(box);
}

void OCTree::Update(vector<AABB3>& boxes, vector<OCTnode*>& nodes,int numtri,bool first)
{
	if(first)
	{
		for (int i = 0; i < numtri; i++)
		{
			if (!root->InsertAABB(boxes, nodes, i))
				nodes[i] = nullptr;
		}
	}else
	{
		for (int i = 0; i < numtri; i++)
		{
			if(nodes[i]==nullptr)
				root->InsertAABB(boxes, nodes, i);
			else
				nodes[i]->UpdateTri(boxes, nodes, false, i);
		}
	}
}

void OCTree::PostDelete(OCTnode* root)
{
	if(root==nullptr)
		return;

	for (int i = 0; i < 8; i++)
	{
		PostDelete(root->_child[i]);
	}
	delete root;
}