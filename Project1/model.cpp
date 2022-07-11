#include "model.h"


#if VERSION == 0


GPULINE Model::Model(const char* path, const char* material_path)
{

	warn = new string();
	err = new string();
	attrib = new tinyobj::attrib_t();
	shapes = new vector<tinyobj::shape_t>();
	material = new vector<tinyobj::material_t>();

	tinyobj::LoadObj(attrib, shapes, material, warn, err, path, material_path);
	for (int i = 0; i < attrib->vertices.size(); i+=3)
	{
		box += Vec3(attrib->vertices[i], attrib->vertices[i + 1], attrib->vertices[i + 2]);
	}
	

#if USEGPU==true
	set<int> lmi;
	string gang = "/";
	int texnum = 0;
	for (int i = 0; i < material->size(); i++)
	{
		tinyobj::material_t& mater = (*material)[i];
		if(mater.diffuse_texname!="")
		{
			texnum++;
			texture_nemes.push_back(material_path + gang + mater.diffuse_texname);
			if (mater.unknown_parameter.find("Le") != mater.unknown_parameter.end())
			{
				auto value = mater.unknown_parameter["Le"];
				stringstream ss;
				ss << value;
				double v1, v2, v3;
				ss >> v1 >> v2 >> v3;

				materials.push_back(Material(Vec3(mater.diffuse[0], mater.diffuse[1], mater.diffuse[2]), Vec3(mater.specular[0], mater.specular[1], mater.specular[2]), mater.shininess, Vec3(v1, v2, v3), 1));
				lmi.insert(i);
			}
			else
			{
				if (mater.shininess > 1)
					materials.push_back(Material(Vec3(mater.diffuse[0], mater.diffuse[1], mater.diffuse[2]), Vec3(mater.specular[0], mater.specular[1], mater.specular[2]), mater.shininess, 2,texnum));
				else
					materials.push_back(Material(Vec3(mater.diffuse[0], mater.diffuse[1], mater.diffuse[2]), Vec3(mater.specular[0], mater.specular[1], mater.specular[2]), mater.shininess, 0, texnum));
			}
		}else
		{
			if (mater.unknown_parameter.find("Le") != mater.unknown_parameter.end())
			{
				auto value = mater.unknown_parameter["Le"];
				stringstream ss;
				ss << value;
				double v1, v2, v3;
				ss >> v1 >> v2 >> v3;

				materials.push_back(Material(Vec3(mater.diffuse[0], mater.diffuse[1], mater.diffuse[2]), Vec3(mater.specular[0], mater.specular[1], mater.specular[2]), mater.shininess, Vec3(v1, v2, v3), 1));
				lmi.insert(i);
			}
			else
			{
				if (mater.shininess > 1)
					materials.push_back(Material(Vec3(mater.diffuse[0], mater.diffuse[1], mater.diffuse[2]), Vec3(mater.specular[0], mater.specular[1], mater.specular[2]), mater.shininess, 2, -1));
				else
					materials.push_back(Material(Vec3(mater.diffuse[0], mater.diffuse[1], mater.diffuse[2]), Vec3(mater.specular[0], mater.specular[1], mater.specular[2]), mater.shininess, 0, -1));
			}
		}
		
	}
	for (int i = 0; i < shapes[0][0].mesh.material_ids.size(); i++)
	{
		if(lmi.find(shapes[0][0].mesh.material_ids[i])!=lmi.end())
		{
			lights.push_back(i);
		}
	}
#else

	for(int i = 0; i < material->size(); i++)
	{
		tinyobj::material_t& mater = (*material)[i];
		if (mater.unknown_parameter.find("Le") != mater.unknown_parameter.end())
		{
			auto value = mater.unknown_parameter["Le"];
			stringstream ss;
			ss << value;
			double v1, v2, v3;
			ss >> v1 >> v2 >> v3;
			materials.push_back(make_shared<Diffuse_Light>(Vec3(mater.diffuse[0], mater.diffuse[1], mater.diffuse[2]), Vec3(mater.specular[0], mater.specular[1], mater.specular[2]), Vec3(v1, v2, v3), mater.shininess));
		}
		else
		{
			materials.push_back(make_shared<Material>(Vec3(mater.diffuse[0], mater.diffuse[1], mater.diffuse[2]), Vec3(mater.specular[0], mater.specular[1], mater.specular[2]), mater.shininess));
		}
	}
#endif
	
	delete warn;
	delete err;
}

GPULINE Model::~Model()
{
	delete material;
	delete attrib;
	delete shapes;
}

#if USEGPU==false

GPULINE bool Model::hit(double t_min, double t_max, const Ray& r, hit_record& record)const
{
	double t = t_max, mint = t_max;
	Vec3 normal, point, center;
	for (int j = 0; j < (*shapes).size(); j++)
	{
		std::vector<tinyobj::index_t>& indices = (*shapes)[j].mesh.indices;
		for (int i = 0, k = 0; i < indices.size(); i += (*shapes)[j].mesh.num_face_vertices[k], k++)
		{
			if ((*shapes)[j].mesh.num_face_vertices[k] == 3 && RayTriangleIntersect(r, t, indices[i], indices[i + 1], indices[i + 2], normal, point, center))
			{
				if (t > t_min && t < t_max && t < mint)
				{
					mint = t;
					record.t = mint;
					record.point = point;
					record.normal = unit_vector(normal);
					record.u = center.x * (*attrib).texcoords[indices[i].texcoord_index * 2] + center.y * (*attrib).texcoords[indices[i + 1].texcoord_index * 2] + center.z * (*attrib).texcoords[indices[i + 2].texcoord_index * 2];
					record.v = center.x * (*attrib).texcoords[indices[i].texcoord_index * 2 + 1] + center.y * (*attrib).texcoords[indices[i + 1].texcoord_index * 2 + 1] + center.z * (*attrib).texcoords[indices[i + 2].texcoord_index * 2 + 1];
					record.color = center.x * Vec3((*attrib).colors[indices[i].vertex_index * 3], (*attrib).colors[indices[i].vertex_index * 3 + 1], (*attrib).colors[indices[i].vertex_index * 3 + 2])
						+ center.y * Vec3((*attrib).colors[indices[i + 1].vertex_index * 3], (*attrib).colors[indices[i + 1].vertex_index * 3 + 1], (*attrib).colors[indices[i + 1].vertex_index * 3 + 2])
						+ center.z * Vec3((*attrib).colors[indices[i + 2].vertex_index * 3], (*attrib).colors[indices[i + 2].vertex_index * 3 + 1], (*attrib).colors[indices[i + 2].vertex_index * 3 + 2]);

					record.mat_ptr = materials[(*shapes)[j].mesh.material_ids[k]];
				}
			}
		}
	}

	if (mint > t_min && mint < t_max)
		return true;
	else
		return false;
}
#endif


GPULINE bool Model::RayTriangleIntersect(const Ray& r, double& t, const tinyobj::index_t& n0, const tinyobj::index_t& n1, const tinyobj::index_t& n2, Vec3& normal, Vec3& Point, Vec3& center)const
{
	Vec3 v0((*attrib).vertices[n0.vertex_index * 3], (*attrib).vertices[n0.vertex_index * 3 + 1], (*attrib).vertices[n0.vertex_index * 3 + 2]);
	Vec3 v1((*attrib).vertices[n1.vertex_index * 3], (*attrib).vertices[n1.vertex_index * 3 + 1], (*attrib).vertices[n1.vertex_index * 3 + 2]);
	Vec3 v2((*attrib).vertices[n2.vertex_index * 3], (*attrib).vertices[n2.vertex_index * 3 + 1], (*attrib).vertices[n2.vertex_index * 3 + 2]);
	Vec3 no0((*attrib).normals[n0.normal_index * 3], (*attrib).normals[n0.normal_index * 3 + 1], (*attrib).normals[n0.normal_index * 3 + 2]);
	Vec3 no1((*attrib).normals[n1.normal_index * 3], (*attrib).normals[n1.normal_index * 3 + 1], (*attrib).normals[n1.normal_index * 3 + 2]);
	Vec3 no2((*attrib).normals[n2.normal_index * 3], (*attrib).normals[n2.normal_index * 3 + 1], (*attrib).normals[n2.normal_index * 3 + 2]);

	bool isIn = false;
	Vec3 E1 = v1 - v0;
	Vec3 E2 = v2 - v0;
	Vec3 S = r.origin - v0;
	Vec3 S1 = cross(r.direction, E2);
	Vec3 S2 = cross(S, E1);
	float coeff = 1.0 / dot(S1, E1);
	float tt = coeff * dot(S2, E2);
	float b1 = coeff * dot(S1, S);
	float b2 = coeff * dot(S2, r.direction);
	if (t >= 0 && b1 >= 0 && b2 >= 0 && (1 - b1 - b2) >= 0)
	{
		isIn = true;
		t = tt;
		center.x = b1;
		center.y = b2;
		center.z = 1 - b1 - b2;
		Point = r.origin + t * r.direction;
		normal = center.x * no0 + center.y * no1 + center.z * no2;
	}
	return isIn;
}


#else


void Model::SetABC(float zFar, float zNear)
{
	ac = -4 / (zFar - zNear) * zFar * zNear / (zFar - zNear);
	_b = (zFar + zNear) / (zFar - zNear);
}

Model::Model(const char* path)
{
	shared_ptr<Mesh> object;
	loadModel(path, object);
	meshes.push_back(object);
	first = true;
	biasv.push_back(0);
	biasf.push_back(0);
	numvtx = object->numvtx;
	numtri = object->numtri;
	face = vector<Triangle>(numtri);
	newz = vector<float>(numvtx);
	nowvertex = vector<Vec3>(numvtx);
	culledface = vector<bool>(numtri);
	culledface2 = vector<bool>(numtri);
	boxes = vector<AABB3>(numtri);
	nodes = vector<OCTnode*>(numtri);
	useTree = false;
	Updateculled = false;
	deleteTree = false;
}

void Model::Draw(Shader* shader, bool& changed)
{
	if (changed)
	{
		shader->HZB->reset();
		SetBox(shader->width, shader->height);
		if (useTree)
		{
			oct.Update(boxes, nodes, numtri, first);
			deleteTree = true;
			if (first)	first = false;
			drawNode(oct.root, shader, changed);
		}
		else
		{
			int count = 0;
			for (int i = 0; i < numtri; i++) {
				if (culledface[i])
					culledface2[i] = shader->drawTriangle(nowvertex[face[i].vid[0]], nowvertex[face[i].vid[1]], nowvertex[face[i].vid[2]],
						boxes[i], Color(rand() % 255, rand() % 255, rand() % 255), true);
				if (!culledface2[i])
					count++;
			}
			for (int i = 0; i < numtri; i++) {
				if (!culledface[i])
				{
					culledface2[i] = shader->drawTriangle(nowvertex[face[i].vid[0]], nowvertex[face[i].vid[1]], nowvertex[face[i].vid[2]],
						boxes[i], Color(rand() % 255, rand() % 255, rand() % 255), true);
				}
			}
			Updateculled = true;
		}
		changed = false;
	}
	else
	{
		if (useTree)
		{
			if (first)
			{
				oct.Update(boxes, nodes, numtri, first);
				first = false;
			}
			if (deleteTree)
			{
				oct.PostdeleteNode(oct.root);
				deleteTree = false;
				drawNode2(oct.root, shader, changed);
			}
			else
			{
				drawNode(oct.root, shader, changed);
			}
		}
		else
		{
			if (Updateculled)
			{
				for (int i = 0; i < numtri; i++) {
					culledface[i] = shader->drawTriangle(nowvertex[face[i].vid[0]], nowvertex[face[i].vid[1]], nowvertex[face[i].vid[2]],
						boxes[i], Color(rand() % 255, rand() % 255, rand() % 255), false);
				}
				Updateculled = false;
			}
			else
			{
				for (int i = 0; i < numtri; i++) {
					if (culledface[i])
						culledface2[i] = shader->drawTriangle(nowvertex[face[i].vid[0]], nowvertex[face[i].vid[1]], nowvertex[face[i].vid[2]],
							boxes[i], Color(rand() % 255, rand() % 255, rand() % 255), false);
				}
				for (int i = 0; i < numtri; i++) {
					if (!culledface[i])
					{
						culledface2[i] = shader->drawTriangle(nowvertex[face[i].vid[0]], nowvertex[face[i].vid[1]], nowvertex[face[i].vid[2]],
							boxes[i], Color(rand() % 255, rand() % 255, rand() % 255), false);
					}
				}
				culledface = culledface2;
			}
		}
	}

}

void Model::DrawTree(Shader* shader)
{
	SetBox(shader->width, shader->height);
	oct.Update(boxes, nodes, numtri, first);

	if (first)	first = false;
	drawNode(oct.root, shader, true);
}

void Model::transfer(unsigned index, const mat4& viewport, const mat4& projection, const mat4& view, const mat4& model)
{
	if (index >= meshes.size())
		return;

	mat4 trans = viewport * projection * view * model;

	int minp = biasv[index], maxp = biasv[index] + meshes[index]->numvtx;
	int j = 0;
	for (int i = minp; i < maxp; i++)
	{
		trans.dotV3(meshes[index]->vertex[j++], nowvertex[i]);
	}
}

void Model::transfer2(unsigned index, const mat4& viewport, const mat4& projection, const mat4& view, const mat4& model,
	float near, float far)
{
	if (index >= meshes.size())
		return;

	mat4 trans = view * model;
	mat4 trans2 = viewport * projection;
	mat4 trans3 = trans2 * trans;
	double a = -2 / (far - near), b = (far + near) / (far - near);
	double c = -2 * far * near / (far - near);
	ac = 2 / (far - near) * -2 * far * near / (far - near);
	b = (far + near) / (far - near);
	int minp = biasv[index], maxp = biasv[index] + meshes[index]->numvtx;
	int j = 0;
	float nz;
	for (int i = minp; i < maxp; i++)
	{
		trans.dotV3(meshes[index]->vertex[j++], nowvertex[i]);
		nz = a * nowvertex[i].z + b;
		trans2.dotV3(nowvertex[i], nowvertex[i]);
		newz[i] = nowvertex[i].z;
		nowvertex[i].z = nz;
	}
}

void Model::addModel(string path)
{
	shared_ptr<Mesh> object;
	loadModel(path, object);
	meshes.push_back(object);

	biasv.push_back(numvtx);
	biasf.push_back(numtri);
	numvtx += object->numvtx;
	numtri += object->numtri;
	face = vector<Triangle>(numtri);
	nowvertex = vector<Vec3>(numvtx);
	culledface = vector<bool>(numtri);
	culledface2 = vector<bool>(numtri);
	newz = vector<float>(numvtx);
	boxes = vector<AABB3>(numtri);
	nodes = vector<OCTnode*>(numtri);
}

void Model::InitOct(int H, int W)
{
	oct.createTree(AABB3(Vec3(0, 0, -1), Vec3(W - 1, H - 1, 1.0f)));
}

void Model::SetBox(int width, int height)
{
	float minz, maxz;
	for (int i = 0; i < numtri; i++) {
		if (nowvertex[face[i].vid[0]].z > nowvertex[face[i].vid[1]].z)
		{
			minz = nowvertex[face[i].vid[1]].z;
			maxz = nowvertex[face[i].vid[0]].z;
		}
		else
		{
			minz = nowvertex[face[i].vid[0]].z;
			maxz = nowvertex[face[i].vid[1]].z;
		}
		if (maxz < nowvertex[face[i].vid[2]].z)
		{
			maxz = nowvertex[face[i].vid[2]].z;
		}
		else if (minz > nowvertex[face[i].vid[2]].z)
		{
			minz = nowvertex[face[i].vid[2]].z;
		}
		boxes[i].set(nowvertex[face[i].vid[0]]);
		boxes[i] += nowvertex[face[i].vid[1]];
		boxes[i] += nowvertex[face[i].vid[2]];
		boxes[i].setDoubleZ(ac / (minz + _b) + _b, ac / (maxz + _b) + _b);
		boxes[i].clamp(Vec3(0, 0, -1.0f), Vec3(width - 1, height - 1, 1.0f));
	}
}

void Model::SetFace()
{
	int bias2 = 0;
	for (int j = 0; j < meshes.size(); j++)
	{
		for (int i = 0; i < meshes[j]->numtri; i++)
		{
			face[i + bias2].setv(meshes[j]->face[i].vid[0] + biasv[j], meshes[j]->face[i].vid[1] + biasv[j], meshes[j]->face[i].vid[2] + biasv[j]);
		}
		bias2 += meshes[j]->numtri;
	}
}

void Model::Switch()
{
	useTree = !useTree;
}

void Model::loadModel(string filename, shared_ptr<Mesh>& object)
{
	int i = 0;
	fstream file;
	file.open(filename);
	if (!file.is_open())
	{
		printf("Can not read file:%s\n", filename);
		return;
	}
	string line;
	unsigned int numbVtx, numTri, numTex, numNor, numbVtx2, numTri2, numTex2, numNor2, numbInd;
	numbVtx = numTri = numTex = numNor = numbVtx2 = numTri2 = numTex2 = numNor2 = numbInd = 0;
	while (getline(file, line))
	{
		if (line[0] == 'v')
		{
			if (line[1] == ' ')
			{
				numbVtx++;
			}
			else if (line[1] == 't')
			{
				numTex++;
			}
			else if (line[1] == 'n')
			{
				numNor++;
			}
		}
		else if (line[0] == 'f')
		{
			numTri++;
		}
	}
	file.close();
	file.open(filename);
	Triangle* nface = new Triangle[numTri];
	Vec3* nvertex = new Vec3[numbVtx];
	Vec3* nnormal = new Vec3[numNor];
	Vec2* ntexcord = new Vec2[numTex];
	unsigned int* indices = new unsigned int[numTri * 3];
	while (getline(file, line))
	{
		if (line[0] == 'v')
		{
			const char* st = line.c_str();
			if (line[1] == ' ')
			{
				double x, y, z;
				sscanf(st + 2, "%lf%lf%lf", &x, &y, &z);
				nvertex[numbVtx2++].set(x, y, z);
			}
			else if (line[1] == 't')
			{
				double x, y;
				sscanf(st + 2, "%lf%lf", &x, &y);
				ntexcord[numTex2++].set(x, y);
			}
			else if (line[1] == 'n')
			{
				double x, y, z;
				sscanf(st + 2, "%lf%lf%lf", &x, &y, &z);
				nnormal[numNor2++].set(x, y, z);
			}
		}
		else if (line[0] == 'f' && line[1] == ' ')
		{
			const char* st = line.c_str();
			int id1, id2, id3, id4, id5, id6, id7, id8, id9;
			const char* nxt;
			sscanf(st + 2, "%d", &id1);
			nxt = strchr(st + 2, ' ');

			sscanf(nxt + 1, "%d", &id4);
			nxt = strchr(nxt + 1, ' ');
			sscanf(nxt + 1, "%d", &id7);

			id1 = id1 < 0 ? (int)numbVtx + id1 : id1 - 1;
			id4 = id4 < 0 ? (int)numbVtx + id4 : id4 - 1;
			id7 = id7 < 0 ? (int)numbVtx + id7 : id7 - 1;
			//sscanf(st + 2, "%u/%u/%u %u/%u/%u %u/%u/%u", &id1, &id2, &id3, &id4, &id5, &id6, &id7, &id8, &id9);
			//indices[numbInd++] = id1 - 1;
			//indices[numbInd++] = id4 - 1;
			//indices[numbInd++] = id7 - 1;
			nface[numTri2++].setv(id1, id4, id7);
			//nface[numTri2].sett(id2 - 1, id5 - 1, id8 - 1);
			//nface[numTri2++].setn(id3 - 1, id6 - 1, id9 - 1);
		}
	}
	object = make_shared<Mesh>(numbVtx2, numTri2, numbInd, nface, nvertex, nnormal, ntexcord);
}

bool Model::drawNode(OCTnode* root, Shader* shader, bool changed)
{
	if (root == nullptr)
		return false;
	if (!shader->HZB->Culling4(root->box, true))
	{
		//printf("box culled!\n");
		return false;
	}
	int i = 0;
	root->Draw(culledface, culledface2, nowvertex, face, shader, newz, boxes, changed);
	for (; i < 4; i++)
	{
		drawNode(root->_child[i], shader, changed);
	}
	for (; i < 8; i++)
	{
		drawNode(root->_child[i], shader, changed);
	}
	return true;
}

bool Model::drawNode2(OCTnode* root, Shader* shader, bool changed)
{
	if (root == nullptr)
		return false;
	if (!shader->HZB->Culling4(root->box, true))
	{
		return false;
	}
	int i = 0;
	root->Draw2(culledface, culledface2, nowvertex, face, shader, newz, boxes, changed);
	for (; i < 8; i++)
	{
		drawNode2(root->_child[i], shader, changed);
	}
	return true;

}
#endif
