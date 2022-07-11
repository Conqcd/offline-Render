#include "hierarchyZbuffer.h"

void HierarchyZbuffer::SetABC(float zFar, float zNear)
{
	ac = 2 / (zFar - zNear) * -2 * zFar * zNear / (zFar - zNear);
	_b = (zFar + zNear) / (zFar - zNear);
}

void HierarchyZbuffer::init(int w, int h)
{
	width = w;
	height = h;
	if(w==1&&h==1)
	{
		HZB.push_back(new ZBuffer(1, 1));
		for (int i = 0; i < height; i++)
		{
			HZB[0]->y_m[i] = i;
		}
		for (int i = 0; i < width; i++)
		{
			HZB[0]->x_m[i] = i;
		}
	}
	else
	{
		int level = 0;
		while (w != 1 || h != 1)
		{
			HZB.push_back(new ZBuffer(w, h));
			if (level > 0)
				HZB[level]->setXY(HZB[level - 1]->width, HZB[level - 1]->height,HZB[level-1]->x_m,HZB[level-1]->y_m);
			else
			{
				for (int i = 0; i < height; i++)
				{
					HZB[level]->y_m[i] = i;
				}
				for (int i = 0; i < width; i++)
				{
					HZB[level]->x_m[i] = i;
				}
			}
			w = w == 1 ? 1 : (w + 1) / 2;
			h = h == 1 ? 1 : (h + 1) / 2;
			level++;
		}
		HZB.push_back(new ZBuffer(1, 1));
		HZB[level]->setXY(HZB[level - 1]->width, HZB[level - 1]->height, HZB[level - 1]->x_m, HZB[level - 1]->y_m);
	}
	
	que = new ZQueue(width * height);
}

bool HierarchyZbuffer::LessThanZ(float z_value, int x, int y)
{
    if (z_value >= HZB[0]->zbuffer[y][x])
    {
        HZB[0]->zbuffer[y][x] = z_value;
        return true;
    }
    else
        return false;
}

void HierarchyZbuffer::Update(int x, int y)
{
	int level = 0;
	while (level<HZB.size()-1)
	{
		int x1, x2, x3, y1, y2, y3;
		findsibling(x, y, x1, y1, x2, y2, x3, y3, HZB[level]->width, HZB[level]->height);
		float z, z1, z2, z3, minz;
		z = HZB[level]->zbuffer[y][x]; 
		z1 = HZB[level]->zbuffer[y1][x1];
		z2 = HZB[level]->zbuffer[y2][x2];
		z3 = HZB[level]->zbuffer[y3][x3];
		minz = fmin(z, z1);
		minz = fmin(minz, z2);
		minz = fmin(minz, z3);
		int pax = x / 2, pay = y / 2;
		float paz = HZB[level + 1]->zbuffer[pay][pax];
		if(minz>paz)
		{
			HZB[level + 1]->zbuffer[pay][pax] = minz;
		}else
		{
			break;
		}
		level++;
		x = pax;
		y = pay;
	}
}

void HierarchyZbuffer::Update2(int xl, int yl,int xb,int yb)
{
	int xm = 615, xM = 645, ym = 12, yM = 61;
	for (int level =0; level <HZB.size()-1; level++)
	{

	//#pragma omp parallel for
		for (int x= xl; x <= xb; x += 2) {
			for (int y = yl; y <= yb; y += 2) {

				if(x>=xm&&x<=xM&&y>=ym&&y<=yM)
				{
					int k = 1;
					k++;
				}
				int x1, x2, x3, y1, y2, y3;
				findsibling(x, y, x1, y1, x2, y2, x3, y3, HZB[level]->width, HZB[level]->height);
				float z, z1, z2, z3, minz;
				z = HZB[level]->zbuffer[y][x];
				z1 = HZB[level]->zbuffer[y1][x1];
				z2 = HZB[level]->zbuffer[y2][x2];
				z3 = HZB[level]->zbuffer[y3][x3];
				minz = fmin(z, z1);
				minz = fmin(minz, z2);
				minz = fmin(minz, z3);
				int pax = x / 2, pay = y / 2;
				float paz = HZB[level + 1]->zbuffer[pay][pax];
				if (minz > paz)
				{
					HZB[level + 1]->zbuffer[pay][pax] = minz;
				}
				
				if (y == yb - 1)
					y--;
			}
			if (x == xb - 1)
				x--;
		}
		xl /= 2;
		xb /= 2;
		yl /= 2;
		yb /= 2;
	}
}

bool HierarchyZbuffer::Culling(const AABB3& box)
{
	bool judge = false;
	float nearest = box._max.z;
	int level = HZB.size() - 1;
	DownNode* last, * last2;
	int count = 0;
	if (Contain(box._min.x, box._min.y, box._max.x, box._max.y, 0, 0, width - 1, height - 1))
	{
		DownNode* dn = new DownNode(level, box._min.x, box._min.y, box._max.x, box._max.y, 0,0 , width - 1, height - 1, 0, 0);
		que->push(dn);
		last2 = last = dn;
	}
	else
	{
		return false;
	}
	bool LevelNotPass = false;
	while (!que->empty())
	{
		DownNode* temp = que->front();
		que->pop();
		count++;

		if (nearest > HZB[temp->level]->zbuffer[temp->indexy][temp->indexx])
		{
			if (temp->level != 0)
			{
				int x1, y1, x2, y2, x3, y3, x4, y4;
				findson(temp->indexx, temp->indexy, x1, y1, x2, y2, x3, y3, x4, y4, HZB[temp->level - 1]->width, HZB[temp->level - 1]->height);
				int midW = (temp->Bmaxx + temp->Bminx) / 2, midH = (temp->Bminy + temp->Bmaxy) / 2;
				if (Contain(temp->Aminx, temp->Aminy, temp->Amaxx, temp->Amaxy, temp->Bminx, temp->Bminy, midW, midH))
				{
					DownNode* dn = new DownNode(temp->level - 1, temp->Aminx, temp->Aminy, midW>temp->Amaxx?temp->Amaxx:midW, midH > temp->Amaxy ? temp->Amaxy : midH,
						temp->Bminx, temp->Bminy, midW, midH, x1, y1);
					que->push(dn);
					last2 = dn;
				}
				if (Contain(temp->Aminx, temp->Aminy, temp->Amaxx, temp->Amaxy, midW + 1, temp->Bminy, temp->Bmaxx, midH))
				{
					int xpo = (midW + 1) < temp->Aminx ? temp->Aminx : (midW + 1);
					xpo = xpo > temp->Amaxx ? temp->Amaxx : xpo;
					DownNode* dn = new DownNode(temp->level - 1, xpo, temp->Aminy, temp->Amaxx, midH > temp->Amaxy ? temp->Amaxy : midH, 
						(midW + 1) > temp->Bmaxx ? temp->Bmaxx : (midW + 1), temp->Bminy, temp->Bmaxx, midH, x2, y2);
					que->push(dn);
					last2 = dn;
				}
				if (Contain(temp->Aminx, temp->Aminy, temp->Amaxx, temp->Amaxy, temp->Bminx, midH + 1, midW, temp->Bmaxy))
				{

					int ypo = (midH + 1) > temp->Aminy ? (midH + 1) : temp->Aminy;
					ypo = ypo > temp->Amaxy ? temp->Amaxy : ypo;
					DownNode* dn = new DownNode(temp->level - 1, temp->Aminx, ypo, midW > temp->Amaxx ? temp->Amaxx : midW, temp->Amaxy, 
						temp->Bminx, (midH + 1) > temp->Bmaxy ? temp->Bmaxy : (midH + 1), midW, temp->Bmaxy, x3, y3);
					que->push(dn);
					last2 = dn;
				}
				if (Contain(temp->Aminx, temp->Aminy, temp->Amaxx, temp->Amaxy, midW + 1, midH + 1, temp->Bmaxx, temp->Bmaxy))
				{

					int xpo = (midW + 1) < temp->Aminx ? temp->Aminx : (midW + 1), ypo = (midH + 1) > temp->Aminy ? (midH + 1) : temp->Aminy;
					xpo = xpo > temp->Amaxx ? temp->Amaxx : xpo;
					ypo = ypo > temp->Amaxy ? temp->Amaxy : ypo;
					DownNode* dn = new DownNode(temp->level - 1, (midW + 1) < temp->Aminx ? temp->Aminx : (midW + 1), (midH + 1) > temp->Aminy ? (midH + 1) : temp->Aminy, temp->Amaxx, temp->Amaxy,
						(midW + 1) > temp->Bmaxx ? temp->Bmaxx : (midW + 1), (midH + 1) > temp->Bmaxy ? temp->Bmaxy : (midH + 1), temp->Bmaxx, temp->Bmaxy, x4, y4);
					que->push(dn);
					last2 = dn;
				}
			}
			LevelNotPass = true;
		}
		if (last == temp)
		{
			level--;
			last = last2;
			if (!LevelNotPass)
			{
				break;
			}
			if (level == 0 && LevelNotPass == true)	judge = true;
			LevelNotPass = false;
		}
		delete temp;
	}
	printf("count=%d\n", count);
	que->clear();
	return judge;
}

bool HierarchyZbuffer::Culling4(const AABB3& box,bool isbox)
{
	bool judge = true;
	float nearest = box._max.z;

	if (isbox)
		nearest = ac / (nearest - _b) - _b;
	int level = HZB.size() - 1;
	int count = 0;
	if (isIn(box._min.x, box._min.y, box._max.x, box._max.y, 0, 0, width - 1, height - 1))
	{

		DownNode* dn = new DownNode(level, box._min.x, box._min.y, box._max.x, box._max.y, 0, 0, width - 1, height - 1, 0, 0);
		que->push(dn);
	}
	else
	{
		return false;
	}
	while (!que->empty())
	{
		DownNode* temp = que->front();
		que->pop();
		//count++;

		if (nearest >= HZB[temp->level]->zbuffer[temp->indexy][temp->indexx])
		{
			if (temp->level != 0)
			{
				int x1, y1, x2, y2, x3, y3, x4, y4;
				findson(temp->indexx, temp->indexy, x1, y1, x2, y2, x3, y3, x4, y4, HZB[temp->level - 1]->width, HZB[temp->level - 1]->height);
				//int midW = (temp->Bmaxx + temp->Bminx) / 2, midH = (temp->Bminy + temp->Bmaxy) / 2;
				int xm, ym, xxm, yym;
				xxm = HZB[temp->level - 1]->x_m[x1];
				yym = HZB[temp->level - 1]->y_m[y1];
				xm = x1 == 0 ? 0 : HZB[temp->level - 1]->x_m[x1 - 1];
				ym = y1 == 0 ? 0 : HZB[temp->level - 1]->x_m[y1 - 1];
				if (isIn(box._min.x, box._min.y, box._max.x, box._max.y, xm, ym, xxm, yym))
				{
					DownNode* dn = new DownNode(temp->level - 1, box._min.x, box._min.y, box._max.x, box._max.y,
						xm, ym, xxm, yym, x1, y1);
					que->push(dn);
				}else
				{
					xxm = HZB[temp->level - 1]->x_m[x2];
					yym = HZB[temp->level - 1]->y_m[y2];
					xm = x2 == 0 ? 0 : HZB[temp->level - 1]->x_m[x2 - 1];
					ym = y2 == 0 ? 0 : HZB[temp->level - 1]->x_m[y2 - 1];
					if (isIn(box._min.x, box._min.y, box._max.x, box._max.y, xm, ym, xxm, yym))
					{
						DownNode* dn = new DownNode(temp->level - 1, box._min.x, box._min.y, box._max.x, box._max.y,
							xm, ym, xxm, yym, x2, y2);
						que->push(dn);
					}else
					{
						xxm = HZB[temp->level - 1]->x_m[x3];
						yym = HZB[temp->level - 1]->y_m[y3];
						xm = x3 == 0 ? 0 : HZB[temp->level - 1]->x_m[x3 - 1];
						ym = y3 == 0 ? 0 : HZB[temp->level - 1]->x_m[y3 - 1];
						if (isIn(box._min.x, box._min.y, box._max.x, box._max.y, xm, ym, xxm, yym))
						{
							DownNode* dn = new DownNode(temp->level - 1, box._min.x, box._min.y, box._max.x, box._max.y,
								xm, ym, xxm, yym, x3, y3);
							que->push(dn);
						}else
						{
							xxm = HZB[temp->level - 1]->x_m[x4];
							yym = HZB[temp->level - 1]->y_m[y4];
							xm = x4 == 0 ? 0 : HZB[temp->level - 1]->x_m[x4 - 1];
							ym = y4 == 0 ? 0 : HZB[temp->level - 1]->x_m[y4 - 1];
							if (isIn(box._min.x, box._min.y, box._max.x, box._max.y, xm, ym, xxm, yym))
							{
								DownNode* dn = new DownNode(temp->level - 1, box._min.x, box._min.y, box._max.x, box._max.y,
									xm, ym, xxm, yym, x4, y4);
								que->push(dn);
							}
						}
					}
				}
			}
		}else
		{
			judge = false;
			delete temp;
			break;
		}
		delete temp;
	}
	//printf("count=%d\n", count);
	que->clear();
	return judge;
}

void HierarchyZbuffer::reset()
{
	for (int i=0;i<HZB.size();i++)
	{
		HZB[i]->reset();
	}
}

void HierarchyZbuffer::PrintZ()
{
	int xm=615, xM=645, ym=12, yM=61;
	for (int i=0;i<HZB.size();i++)
	{
		printf("level:%d\n",i);
		for (int j=0;j<HZB[i]->width;j++)
		{
			for (int k = 0; k < HZB[i]->height; k++)
			{
				/*if (i==8&&(j == 3) && (k==0))
					printf("%f %d %d", HZB[i]->zbuffer[k][j],k,j);*/
				if((j<= xM &&j>= xm)&&(k>= ym &&k<= yM))
					printf("%f %d %d ", HZB[i]->zbuffer[k][j],j,k);
			}
			printf("\n");
		}
		printf("\n");
		xm /= 2;
		xM /= 2;
		ym /= 2;
		yM /= 2;
	}
}

bool HierarchyZbuffer::Culling3(const AABB3& box)
{
	bool judge = false;
	float nearest = box._max.z;
	int level = HZB.size() - 1;
	
	queue<DownNode* > qu;
	DownNode *last,*last2;
	if (Contain(box._min.x, box._min.y, box._max.x, box._max.y,0,0, width - 1,height-1))
	{
		DownNode *dn=new DownNode(level, box._min.x, box._min.y, box._max.x, box._max.y, 0, 0, width - 1, height - 1, 0, 0);
		qu.push(dn);
		last2 = last = dn;
	}else
	{
		return false;
	}
	bool LevelNotPass = false;
	while (!qu.empty())
	{
		DownNode* temp = qu.front();
		qu.pop();
		if (nearest>HZB[temp->level]->zbuffer[temp->indexy][temp->indexx])
		{
			if(temp->level !=0)
			{
				int x1, y1, x2, y2, x3, y3, x4, y4;
				findson(temp->indexx, temp->indexy, x1, y1, x2, y2, x3, y3, x4, y4, HZB[temp->level - 1]->width, HZB[temp->level - 1]->height);
				int midW = (temp->Bmaxx + temp->Bminx) / 2, midH = (temp->Bminy + temp->Bmaxy) / 2;
				if (Contain(temp->Aminx, temp->Aminy, temp->Amaxx, temp->Amaxy, temp->Bminx, temp->Bminy, midW, midH))
				{
					DownNode* dn = new DownNode(temp->level - 1, temp->Aminx, temp->Aminy, midW > temp->Amaxx ? temp->Amaxx : midW, midH > temp->Amaxy ? temp->Amaxy : midH,
						temp->Bminx, temp->Bminy, midW, midH, x1, y1);
					qu.push(dn);
					last2 = dn;
				}
				if (Contain(temp->Aminx, temp->Aminy, temp->Amaxx, temp->Amaxy, midW + 1, temp->Bminy, temp->Bmaxx, midH))
				{
					DownNode* dn = new DownNode(temp->level - 1, (midW + 1) < temp->Aminx ? temp->Aminx : (midW + 1), temp->Aminy, temp->Amaxx, midH > temp->Amaxy ? temp->Amaxy : midH,
						midW + 1, temp->Bminy, temp->Bmaxx, midH, x2, y2);
					qu.push(dn);
					last2 = dn;
				}
				if (Contain(temp->Aminx, temp->Aminy, temp->Amaxx, temp->Amaxy, temp->Bminx, midH + 1, midW, temp->Bmaxy))
				{

					DownNode* dn = new DownNode(temp->level - 1, temp->Aminx, (midH + 1) > temp->Aminy ? (midH + 1) : temp->Aminy, midW > temp->Amaxx ? temp->Amaxx : midW, temp->Amaxy,
						temp->Bminx, midH + 1, midW, temp->Bmaxy, x3, y3);
					qu.push(dn);
					last2 = dn;
				}
				if (Contain(temp->Aminx, temp->Aminy, temp->Amaxx, temp->Amaxy, midW + 1, midH + 1, temp->Bmaxx, temp->Bmaxy))
				{
					DownNode* dn = new DownNode(temp->level - 1, (midW + 1) < temp->Aminx ? temp->Aminx : (midW + 1), (midH + 1) > temp->Aminy ? (midH + 1) : temp->Aminy, temp->Amaxx, temp->Amaxy,
						midW + 1, midH + 1, temp->Bmaxx, temp->Bmaxy, x4, y4);
					qu.push(dn);
					last2 = dn;
				}
			}
			LevelNotPass = true;
		}
		
		if (last==temp)
		{
			level--;
			last = last2;
			if(!LevelNotPass)
			{
				break;
			}
			if (level == 0 && LevelNotPass == true)	judge = true;
			LevelNotPass = false;
		}
		delete temp;
	}
	return judge;
}

bool HierarchyZbuffer::Culling2(const AABB3& box)
{
	bool judge = false;
	float nearest = box._max.z;
	int level = HZB.size() - 1;
	queue<DownNode> qu;
	DownNode last, last2;
	if (Contain(box._min.x, box._min.y, box._max.x, box._max.y, 0, 0, width - 1, height - 1))
	{
		DownNode dn(level, box._min.x, box._min.y, box._max.x, box._max.y, 0, width - 1, 0, height - 1, 0, 0);
		qu.push(dn);
		last = dn;
	}
	else
	{
		return false;
	}
	bool LevelNotPass = false;
	while (!qu.empty())
	{
		DownNode temp = qu.front();
		qu.pop();
		if (nearest > HZB[temp.level]->zbuffer[temp.indexy][temp.indexx])
		{
			if (temp.level != 0)
			{
				int x1, y1, x2, y2, x3, y3, x4, y4;
				findson(temp.indexx, temp.indexy, x1, y1, x2, y2, x3, y3, x4, y4, HZB[temp.level - 1]->width, HZB[temp.level - 1]->height);
				int midW = (temp.Bmaxx + temp.Bminx) / 2, midH = (temp.Bminy + temp.Bmaxy);
				if (Contain(temp.Aminx, temp.Aminy, temp.Amaxx, temp.Amaxy, temp.Bminx, temp.Bminy, midW, midH))
				{
					DownNode dn(temp.level - 1, temp.Aminx, temp.Aminy, midW, midH, temp.Bminx, temp.Bminy, midW, midH, x1, y1);
					qu.push(dn);
					last2 = dn;
				}
				if (Contain(temp.Aminx, temp.Aminy, temp.Amaxx, temp.Amaxy, midW, temp.Bmaxx, temp.Bminy, midH))
				{
					DownNode dn(temp.level - 1, midW, temp.Aminy, temp.Amaxx, midH, midW, temp.Bmaxx, temp.Bminy, midH, x2, y2);
					qu.push(dn);
					last2 = dn;
				}
				if (Contain(temp.Aminx, temp.Aminy, temp.Amaxx, temp.Amaxy, temp.Bminx, midW, midH, temp.Bmaxy))
				{
					DownNode dn(temp.level - 1, temp.Aminx, midH, midW, temp.Amaxy, temp.Bminx, midW, midH, temp.Bmaxy, x3, y3);
					qu.push(dn);
					last2 = dn;
				}
				if (Contain(temp.Aminx, temp.Aminy, temp.Amaxx, temp.Amaxy, midW, midH, temp.Bmaxx, temp.Bmaxy))
				{
					DownNode dn(temp.level - 1, midW, midH, temp.Amaxx, temp.Amaxy, midW, midH, temp.Bmaxx, temp.Bmaxy, x4, y4);
					qu.push(dn);
					last2 = dn;
				}
			}
			LevelNotPass = true;
		}
		if (last == temp)
		{
			level--;
			last = last2;
			if (!LevelNotPass)
			{
				break;
			}
			if (level == 0 && LevelNotPass == true)	judge = true;
			LevelNotPass = false;
		}
	}
	return judge;
}

void HierarchyZbuffer::findsibling(int x, int y, int& x1, int& y1, int& x2, int& y2, int& x3, int& y3,int w,int h)
{
	if (x % 2 == 0)
	{
		x1 = (x + 1) == w ? x : x + 1; x2 = x; x3 = x1;
	}
	else
	{
		x1 = x - 1; x2 = x; x3 = x1;
	}
	if (y % 2 == 0)
	{
		y1 = (y + 1) == h ? y : y + 1; y2 = y1; y3 = y;
	}
	else
	{
		y1 = y - 1; y2 = y1; y3 = y;
	}
}

void HierarchyZbuffer::findson(int x, int y, int& x1, int& y1, int& x2, int& y2, int& x3, int& y3, int& x4, int& y4,
	int w, int h)
{
	x1 = x * 2;
	y1 = y * 2;
	x4 = (x * 2 + 1) == w ? x1 : (x * 2 + 1);
	y4 = (y * 2 + 1) == h ? y1 : (y * 2 + 1);
	x2 = x4;
	y2 = y1;
	x3 = x1;
	y3 = y4;
}

bool HierarchyZbuffer::Contain(int Aminx, int Aminy, int Amaxx, int Amaxy, int Bminx, int Bminy, int Bmaxx, int Bmaxy)
{
	if (Aminx > Bmaxx) return false;
	if (Aminy > Bmaxy) return false;

	if (Amaxx < Bminx) return false;
	if (Amaxy < Bminy) return false;

	return true;
}

bool HierarchyZbuffer::isIn(int Aminx, int Aminy, int Amaxx, int Amaxy, int Bminx, int Bminy, int Bmaxx, int Bmaxy)
{
	if (Amaxx < Aminx || Amaxy < Aminy)
		return false;
	if (Aminx >= Bminx&& Aminy >= Bminy&& Amaxx <= Bmaxx&& Amaxy <= Bmaxy) 
		return true;
	return false;
}