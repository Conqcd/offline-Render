#pragma once

#include"shader.h"
#include <memory>
#include <string>
#include<vector>
#include <set>
#include <fstream>
#include <sstream>

#include "tiny_obj_loader.h"
#include"mesh.h"
#include "material.h"
#include "octree.h"
using namespace std;



#if VERSION == 0

class Model
{
public:
	Model(const char* path, const char* material_path);
	~Model();
	void set_scene_path(const char* path)
	{
		scene_path = path;
	}
#if USEGPU==false
	bool hit(double t_min, double t_max, const Ray& r, hit_record& record)const;
#endif
private:
	bool RayTriangleIntersect(
		const Ray& r, double& t,
		const tinyobj::index_t& v0, const tinyobj::index_t& v1, const tinyobj::index_t& v2,
		Vec3& normal, Vec3& Point, Vec3& center)const;
public:
	AABB3 box;
	tinyobj::attrib_t* attrib;
	vector<tinyobj::shape_t>* shapes;
	string* warn, * err;
	vector<tinyobj::material_t>* material;
	vector<int> lights;
	vector<string> texture_nemes;
	string scene_path="";

#if USEGPU==true
	vector<Material> materials;
#else
	vector<shared_ptr<Material>> materials;
#endif
	/*
	vector<Triangle> world;
	vector<Triangle> lights;

	vector<Vec3> vertex;

	vector<Material> materials;
	*/
};


class Node;
class BVH;
typedef Node* NodePtr;
typedef Node InternalNode;
typedef Node LeafNode;

class GPU_DataSet
{

public:
	GPU_DataSet(const Model& model, int height, int width);

	GPULINE ~GPU_DataSet()
	{
		//deleteMySelf();
	}

	GPULINE bool hit(double t_min, double& t_max, const Ray& r, hit_record& record, int id)const;

	__device__ DoubleColor light_color(hit_record& rec, const Ray& ray) const;
	__device__ Vec3 random(curandState* curand_states, const Vec3& o) const;
	__device__ double pdf_value(const Ray& r, float infinity, bool& needStop, hit_record& record)const;
	
	GPULINE void deleteMySelf();
	__device__ bool bvh_hit(double t_min, double t_max, const Ray& r, hit_record& record)const;
	void load_textures(const Model& model);
#if USEGPU==true
//	__global__ void new_data(double t_min, double t_max, const Ray& r, hit_record& record);
#endif
	GPULINE bool nobvh_hit(double t_min, double t_max, const Ray& r, hit_record& record)const;
	
private:
	GPULINE bool RayTriangleIntersect(
		const Ray& r, double& t, float& area, bool& is_preface,
		const tinyobj::index_t& n0, const tinyobj::index_t& n1, const tinyobj::index_t& n2,
		Vec3& normal, Vec3& Point, Vec3& center)const;

public:
	int matsize;
	int texsize;
	int ligsize;
	cudaTextureObject_t* texid, * dev_texid;
	cudaTextureObject_t scene_id = 0;
	Material* materials;
	int numtri, numvtx;
	float* dev_vtxs;
	float* dev_nmls;
	float* dev_cols;
	float* dev_texs;
	int* dev_matids;
	int* dev_light;
	tinyobj::index_t* dev_ids;
	unsigned long long* dev_morton_code_array;
	unsigned int* dev_index_array;
	LeafNode* leafNodes;
	InternalNode* internalNodes;
	BVH* bvh;
	curandState* dev_states;
	int width, height;
	DoubleColor* dev_image;
	DoubleColor* image;
};

typedef vector<Model> hittable_list;

#else

class Model
{
public:
	vector<shared_ptr<Mesh>> meshes;
	string directory;
	bool first;
	OCTree oct;
	vector<int> biasv, biasf;
	vector<Triangle> face;
	vector<Vec3> nowvertex;
	vector<float> newz;
	unsigned int numvtx, numtri;
	vector<bool> culledface, culledface2;
	vector<AABB3> boxes;
	vector<OCTnode*>nodes;
	bool useTree, deleteTree, Updateculled;
	float ac, _b;
public:

	void SetABC(float zFar, float zNear);
	Model(const char* path);
	void Draw(Shader* shader, bool& changed);
	void DrawTree(Shader* shader);
	void transfer(unsigned int index, const mat4& viewport, const mat4& projection, const mat4& view, const mat4& model);
	void transfer2(unsigned int index, const mat4& viewport, const mat4& projection, const mat4& view, const mat4& model, float near, float far);
	void addModel(string path);
	void InitOct(int H, int W);
	void SetBox(int width, int height);
	void SetFace();
	void Switch();
private:
	void loadModel(string path, shared_ptr<Mesh>& object);
	bool drawNode(OCTnode* root, Shader* shader, bool changed);
	bool drawNode2(OCTnode* root, Shader* shader, bool changed);
};

#endif	


