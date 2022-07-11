#pragma once
#include "box.h"
#include "model.h"

#if USEGPU==true
class Node
{
public:
	Node* parent, * mostRight, * childA, * childB;
	int chiA, chiB;
	AABB3 bounding_box;
	bool isleaf;
	int objectID;
public:
	GPULINE Node(bool leaf)
	{
		isleaf = leaf;
		parent = nullptr;
	}
	GPULINE Node(Node* _childA, Node* _childB, bool leaf)
	{
		isleaf = leaf;
		childA = _childA;
		childB = _childB;
		bounding_box = childA->bounding_box + childB->bounding_box;
		mostRight = nullptr;
	}
	GPULINE Node(int id, bool leaf)
	{
		isleaf = leaf;
		objectID = id;
	}
	GPULINE Node(int* id, bool leaf)
	{
		objectID = *id;
		isleaf = leaf;
	}
	GPULINE Node() = default;
	GPULINE bool hit(double t_min, double& t_max, const Ray& r, hit_record& record, GPU_DataSet* data_set,int depth)const
	{
		if(isleaf)
		{
			return data_set->hit(t_min, t_max, r, record, objectID);
		}else
		{
			if (!bounding_box.hit(t_min, t_max, r))	return false;
			bool hit_left = childA->hit(t_min, t_max, r, record, data_set, depth + 1);
			bool hit_right = childB->hit(t_min, hit_left ? record.t : t_max, r, record, data_set, depth + 1);
			return hit_left || hit_right;
		}
	}
};

typedef Node* NodePtr;
typedef Node InternalNode;
typedef Node LeafNode;


class BVH
{
public:
	NodePtr root;
	int numObjects;
	NodePtr LeavesPtr;
public:
	GPULINE BVH() = default;
	GPULINE ~BVH()
	{

	}
	GPULINE bool hit(double t_min, double t_max, const Ray& r, hit_record& record,GPU_DataSet* data_set)const
	{
		return root->hit(t_min, t_max, r, record, data_set,0);
	}
	
	GPULINE int getObjectIdx(NodePtr node)
	{
		return node->objectID;
	}

	GPULINE bool isLeaf(NodePtr node)
	{
		return node->isleaf;
	}

	GPULINE NodePtr getRightmostLeafInLeftSubtree(NodePtr node)
	{
		if (node->isleaf)
			return node;
		if (node->childA->mostRight != nullptr)
		{
			return node->childA->mostRight;
		}
		else
		{
			NodePtr ptr = node->childA;
			while (!ptr->isleaf)
			{
				ptr = ptr->childB;
			}
			node->childA->mostRight = ptr;
			return ptr;
		}
	}

	GPULINE NodePtr getRightmostLeafInRightSubtree(NodePtr node)
	{
		if (node->isleaf)
			return node;
		if (node->childB->mostRight != nullptr)
		{
			return node->childB->mostRight;
		}
		else
		{
			NodePtr ptr = node->childB;
			while (!ptr->isleaf)
			{
				ptr = ptr->childB;
			}
			node->childB->mostRight = ptr;
			return ptr;
		}
	}

	GPULINE int getNumLeaves()
	{
		return numObjects;
	}

	GPULINE NodePtr getRoot()
	{
		return root;
	}

	GPULINE NodePtr getLeftChild(NodePtr node)
	{
		return node->childA;
	}

	GPULINE NodePtr getRightChild(NodePtr node)
	{
		return node->childB;
	}

	GPULINE AABB3 getAABB(NodePtr node)
	{
		return node->bounding_box;
	}

	GPULINE NodePtr getLeaf(int idx)
	{
		return &LeavesPtr[idx];
	}
};

#endif
