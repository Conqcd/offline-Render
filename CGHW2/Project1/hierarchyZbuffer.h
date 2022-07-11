#pragma once
#include "box.h"
#include <vector>
#include <queue>
using namespace std;

struct DownNode
{
	int level;
	int Aminx, Aminy, Amaxx, Amaxy, Bminx, Bminy, Bmaxx, Bmaxy;
	int indexx, indexy;

	DownNode()
	{}
	DownNode(int l,int _mx,int _my,int _Mx,int _My, int _mx2, int _my2, int _Mx2, int _My2,int x,int y):level(l), Aminx(_mx), Aminy(_my),
	Amaxx(_Mx), Amaxy(_My), Bminx(_mx2), Bminy(_my2), Bmaxx(_Mx2), Bmaxy(_My2),indexx(x),indexy(y)
	{
		
	}
	
	void operator=(const DownNode& rhs)
	{
		level=rhs.level;
		Aminx = rhs.Aminx;
		Aminy = rhs.Aminy;
		Amaxx = rhs.Amaxx;
		Amaxy = rhs.Amaxy;
		Bminx = rhs.Bminx;
		Bminy = rhs.Bminy;
		Bmaxx = rhs.Bmaxx;
		Bmaxy = rhs.Bmaxy;
		indexx = rhs.indexx;
		indexy = rhs.indexy;
	}
	bool operator==(const DownNode& rhs)
	{
		if (level == rhs.level && Aminx == rhs.Aminx && Aminy == rhs.Aminy && Amaxx == rhs.Amaxx && Amaxy == rhs.Amaxy
			&& Bminx == rhs.Bminx && Bminy == rhs.Bminy && Bmaxx == rhs.Bmaxx && Bmaxy == rhs.Bmaxy
			&& indexx == rhs.indexx && indexy == rhs.indexy)
			return true;
		return false;
	}
};

class ZQueue
{
public:
	DownNode** point;
	int head;
	int tail;
	int capacity;
public:
	ZQueue(int number)
	{
		point = new DownNode * [number+1];
		head = tail = 0;
		capacity = number + 1;
	}

	bool empty()
	{
		return head == tail;
	}

	void push(DownNode* p)
	{
		point[head] = p;
		head = (head + 1) % capacity;
	}

	void pop()
	{
		tail = (tail + 1) % capacity;
	}

	DownNode* front()
	{
		return point[tail];
	}

	void clear()
	{
		while (tail!=head)
		{
			delete point[tail];
			tail = (tail + 1) % capacity;
		}
	}
	
	~ZQueue()
	{
		delete[]point;
	}
};

class ZBuffer
{
public:
	int width, height;
	float** zbuffer;
	int* x_m, * y_m;
public:

	ZBuffer(int w=800, int h=600) :width(w), height(h)
	{
		zbuffer = new float*[height];
		x_m = new int[width];
		y_m = new int[height];
		for (int i=0;i<height;i++)
		{
			zbuffer[i] = new float[width];
		}
		reset();
	}
	void reset()
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				zbuffer[i][j] = -FLT_MAX;
			}
		}
	}
	void setXY(int widthp,int heightp,int *xx,int *yy)
	{
		int position;
		for (int i = 0; i < height; i++)
		{
			position = (i * 2 + 1);
			position = position >= heightp ? heightp - 1 : position;
			y_m[i] = yy[position];
		}
		for (int i = 0; i < width; i++)
		{
			position = (i * 2 + 1);
			position = position >= widthp ? widthp - 1 : position;
			x_m[i] = xx[position];
		}
	}
	~ZBuffer()
	{
		for (int i = 0; i < height; i++)
		{
			delete[] zbuffer[i];
		}
		delete[]x_m;
		delete[]y_m;
	}
};

class HierarchyZbuffer 
{
public:
	int width, height;
	vector<ZBuffer*> HZB;
	ZQueue* que;
	float ac, _b;
public:
	HierarchyZbuffer(int w = 800, int h = 600):width(w),height(h)
	{
		init(w, h);
	}
	void SetABC(float zFar,float zNear);
	void init(int w, int h);
	bool LessThanZ(float z_value,int x,int y);
	void Update(int x,int y);
	void Update2(int x1, int y1, int x2, int y2);
	bool Culling(const AABB3& box);
	bool Culling2(const AABB3& box);
	bool Culling3(const AABB3& box);
	bool Culling4(const AABB3& box,bool isbox);
	void reset();
	//ต๗สิ
	void PrintZ();
	~HierarchyZbuffer()
	{
		for (int i=0;i<HZB.size();i++)
		{
			delete HZB[i];
		}
		delete que;
	}
private:
	void findsibling(int x,int y,int &x1,int &y1, int& x2, int& y2, int& x3, int& y3,int w,int h);
	void findson(int x, int y, int& x1, int& y1, int& x2, int& y2, int& x3, int& y3, int& x4, int& y4, int w, int h);
	bool Contain(int Aminx,int Aminy, int Amaxx, int Amaxy, int Bminx, int Bminy, int Bmaxx, int Bmaxy);
	bool isIn(int Aminx, int Aminy, int Amaxx, int Amaxy, int Bminx, int Bminy, int Bmaxx, int Bmaxy);
};