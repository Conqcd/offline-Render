#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/copy.h>
#include<thrust/sort.h>
#include<thrust/execution_policy.h>

#include "bvh.h"
#include "model.h"
#include "book.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

__device__ unsigned int __clz_long(unsigned long long v)
{
	unsigned v1, v2;
	v1 = v;
	v2 = v >> 32;
	if(v2!=0)
	{
		return __clz(v2);
	}else
	{
		return 32 + __clz(v1);
	}
}

__device__ unsigned long long expandBits_long(unsigned long long v)
{
	v = (v * 0x0000000100000001llu) & 0xFFFF00000000FFFFllu;
	v = (v * 0x00010001llu) & 0x00FF0000FF0000FFllu;
	v = (v * 0x00000101llu) & 0xF00F00F00F00F00Fllu;
	v = (v * 0x00000011llu) & 0x30C30C30C30C30C3llu;
	v = (v * 0x00000005llu) & 0x9249249249249249llu;
	//v = (v * 0x00010001llu) & 0xFF0000FFllu;
	//v = (v * 0x00000101llu) & 0x0F00F00Fllu;
	//v = (v * 0x00000011llu) & 0xC30C30C3llu;
	//v = (v * 0x00000005llu) & 0x49249249llu;
	return v;
}

__device__ unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

__device__ unsigned long long morton3D_long(float x, float y, float z)
{
	//2097152.f
	x = fmin(fmax(x * 2097152.f, 0.0f), 2097151.f);
	y = fmin(fmax(y * 2097152.f, 0.0f), 2097151.f);
	z = fmin(fmax(z * 2097152.f, 0.0f), 2097151.f);
	unsigned long long xx = expandBits_long((unsigned long long)x);
	unsigned long long yy = expandBits_long((unsigned long long)y);
	unsigned long long zz = expandBits_long((unsigned long long)z);
	return xx * 4 + yy * 2 + zz;
}

__device__ unsigned int morton3D(float x, float y, float z)
{
	x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
	y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
	z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int)x);
	unsigned int yy = expandBits((unsigned int)y);
	unsigned int zz = expandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

__global__ void makeMorton(unsigned int Faces, tinyobj::index_t* _tris, float* _vtxs,
                           unsigned int* morton_code_array, unsigned int* index_array,
                           double ww, double hh, double dd, double wb, double hb, double db)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= Faces)
		return;

	double w = 0, h = 0, d = 0;
	for (int k = 0; k < 3; ++k)
	{
		w += _vtxs[_tris[i * 3 + k].vertex_index * 3];
		h += _vtxs[_tris[i * 3 + k].vertex_index * 3 + 1];
		d += _vtxs[_tris[i * 3 + k].vertex_index * 3 + 2];
	}
	w /= 3;
	h /= 3;
	d /= 3;
	morton_code_array[i] = morton3D((w - wb) / ww, (h - hb) / hh, (d - db) / dd);
	//printf("idx:%d %lf %lf %lf %lf %lf %lf\n", i, w, h, d, (w - wb) / ww, (h - hb) / hh, (d - db) / dd);
	index_array[i] = i;
}

__global__ void makeMorton_long(unsigned int Faces, tinyobj::index_t* _tris, float* _vtxs,
	unsigned long long* morton_code_array, unsigned int* index_array,
	double ww, double hh, double dd, double wb, double hb, double db)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= Faces)
		return;

	double w = 0, h = 0, d = 0;
	for (int k = 0; k < 3; ++k)
	{
		w += _vtxs[_tris[i * 3 + k].vertex_index * 3];
		h += _vtxs[_tris[i * 3 + k].vertex_index * 3 + 1];
		d += _vtxs[_tris[i * 3 + k].vertex_index * 3 + 2];
	}
	w /= 3;
	h /= 3;
	d /= 3;
	morton_code_array[i] = morton3D_long((w - wb) / ww, (h - hb) / hh, (d - db) / dd);
	//printf("%llu\n", morton_code_array[i]);
	//printf("idx:%d %lf %lf %lf %lf %lf %lf\n", i, w, h, d, (w - wb) / ww, (h - hb) / hh, (d - db) / dd);
	index_array[i] = i;
}

__device__ long long clz_index_long(long long idx, long long idy, unsigned int NumObjects, unsigned long long* sortedMortonCodes)
{
	return sortedMortonCodes[idx] == sortedMortonCodes[idy]
		? (NumObjects - max(idx, idy)) + 64
		: __clz_long(sortedMortonCodes[idx] ^ sortedMortonCodes[idy]);
}

__device__ int clz_index(int idx, int idy, unsigned int NumObjects, unsigned int* sortedMortonCodes)
{
	return sortedMortonCodes[idx] == sortedMortonCodes[idy]
		       ? (NumObjects - max(idx, idy)) + 32
		       : __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idy]);
}

__device__ int clz_safe(int idx, int idy, unsigned int NumObjects, unsigned int* sortedMortonCodes)
{
	if (idy < 0 || idy > NumObjects - 1) return -1;
	return clz_index(idx, idy, NumObjects, sortedMortonCodes);
}

__device__ long long clz_safe_long(long long idx, long long idy, unsigned int NumObjects, unsigned long long* sortedMortonCodes)
{
	if (idy < 0 || idy > NumObjects - 1) return -1;
	return clz_index_long(idx, idy, NumObjects, sortedMortonCodes);
}

__device__ int findSplit(unsigned int* sortedMortonCodes,int first,int last) {
	unsigned int firstCode = sortedMortonCodes[first];
	unsigned int lastCode = sortedMortonCodes[last];

	if (firstCode == lastCode)
		return (first + last) >> 1;

	int commonPrefix = __clz(firstCode ^ lastCode);

	int split = first;
	int step = last - first;

	do
	{
		step = (step + 1) >> 1;
		int newSplit = split + step;

		if (newSplit < last)
		{
			unsigned int splitCode = sortedMortonCodes[newSplit];
			int splitPrefix = __clz(firstCode ^ splitCode);
			if (splitPrefix > commonPrefix)
				split = newSplit;
		}
	} while (step > 1);
	return split;
}

__device__ long long findSplit_long(unsigned long long* sortedMortonCodes,long long first,long long last) {
	unsigned long long firstCode = sortedMortonCodes[first];
	unsigned long long lastCode = sortedMortonCodes[last];

	if (firstCode == lastCode)
		return (first + last) >> 1;

	long long commonPrefix = __clz_long(firstCode ^ lastCode);

	long long split = first;
	long long step = last - first;

	do
	{
		step = (step + 1) >> 1;
		long long newSplit = split + step;

		if (newSplit < last)
		{
			unsigned long long splitCode = sortedMortonCodes[newSplit];
			long long splitPrefix = __clz_long(firstCode ^ splitCode);
			if (splitPrefix > commonPrefix)
				split = newSplit;
		}
	} while (step > 1);
	return split;
}

__device__ int2 determineRange(unsigned int* sortedMortonCodes, size_t numObjects, unsigned int i)
{
	int d = (clz_safe(i, i + 1, numObjects, sortedMortonCodes) - clz_safe(i, i - 1, numObjects, sortedMortonCodes)) > 0 ? 1 : -1;
	int commonPrefixMin = clz_safe(i, i - d, numObjects, sortedMortonCodes);
	int l_max = 2;
	while (clz_safe(i, i + d * l_max, numObjects, sortedMortonCodes) > commonPrefixMin)
	{
		l_max *= 2;
	}
	int l = 0;
	int t = l_max;
	do
	{
		t = (t + 1) >> 1;
		if (clz_safe(i, i + d * (l + t), numObjects, sortedMortonCodes) > commonPrefixMin)
		{
			l += t;
		}
	} while (t > 1);
	int j = i + l * d;
	int2 range = d > 0 ? make_int2(i, j) : make_int2(j, i);
	return range;
}

__device__ longlong2 determineRange_long(unsigned long long* sortedMortonCodes, size_t numObjects, unsigned int i)
{
	long long d = (clz_safe_long(i, i + 1, numObjects, sortedMortonCodes) - clz_safe_long(i, i - 1, numObjects, sortedMortonCodes)) > 0 ? 1 : -1;
	long long commonPrefixMin = clz_safe_long(i, i - d, numObjects, sortedMortonCodes);
	long long l_max = 2;
	while (clz_safe_long(i, i + d * l_max, numObjects, sortedMortonCodes) > commonPrefixMin)
	{
		l_max *= 2;
	}
	long long l = 0;
	long long t = l_max;
	do
	{
		t = (t + 1) >> 1;
		if (clz_safe_long(i, i + d * (l + t), numObjects, sortedMortonCodes) > commonPrefixMin)
		{
			l += t;
		}
	} while (t > 1);
	long long j = i + l * d;
	longlong2 range = d > 0 ? make_longlong2(i, j) : make_longlong2(j, i);
	return range;
}

__global__ void createNode_long(LeafNode* leafNodes, InternalNode* internalNodes, unsigned long long* sortedMortonCodes,
	unsigned int* sortedObjectIDs, size_t numObjects, BVH* bvh) {
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= numObjects - 1)
		return;
	longlong2 range = determineRange_long(sortedMortonCodes, numObjects, idx);
	long long first = range.x;
	long long last = range.y;


	long long split = findSplit_long(sortedMortonCodes, first, last);

	NodePtr childA;
	if (split == first)
		childA = &leafNodes[split],
		internalNodes[idx].chiA = -1;
	else
		childA = &internalNodes[split],
		internalNodes[idx].chiA = split;

	NodePtr childB;
	if (split + 1 == last)
		childB = &leafNodes[split + 1],
		internalNodes[idx].chiB = -1;
	else
		childB = &internalNodes[split + 1],
		internalNodes[idx].chiB = split + 1;
	internalNodes[idx].childA = childA;
	internalNodes[idx].childB = childB;
	internalNodes[idx].isleaf = false;
	childA->parent = &internalNodes[idx];
	childB->parent = &internalNodes[idx];
	for (long long i = first; i <= last; i++)
	{
		if (i == first)
			internalNodes[idx].bounding_box = leafNodes[i].bounding_box;
		else
			internalNodes[idx].bounding_box += leafNodes[i].bounding_box;
	}
	if (idx == 0)
	{
		bvh->root = &internalNodes[idx];
		bvh->LeavesPtr = leafNodes;
		bvh->numObjects = numObjects;
	}
}

__global__ void createNode(LeafNode* leafNodes, InternalNode* internalNodes, unsigned int* sortedMortonCodes,
                           unsigned int* sortedObjectIDs, size_t numObjects, BVH* bvh) {
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= numObjects - 1)
		return;
	int2 range = determineRange(sortedMortonCodes, numObjects, idx);
	int first = range.x;
	int last = range.y;

	
	int split = findSplit(sortedMortonCodes, first, last);

	NodePtr childA;
	if (split == first)
		childA = &leafNodes[split],
			internalNodes[idx].chiA = -1;
	else
		childA = &internalNodes[split],
			internalNodes[idx].chiA = split;

	NodePtr childB;
	if (split + 1 == last)
		childB = &leafNodes[split + 1],
			internalNodes[idx].chiB = -1;
	else
		childB = &internalNodes[split + 1],
			internalNodes[idx].chiB = split + 1;
	internalNodes[idx].childA = childA;
	internalNodes[idx].childB = childB;
	internalNodes[idx].isleaf = false;
	childA->parent = &internalNodes[idx];
	childB->parent = &internalNodes[idx];
	for (int i = first; i <= last; i++)
	{
		if (i == first)
			internalNodes[idx].bounding_box = leafNodes[i].bounding_box;
		else
			internalNodes[idx].bounding_box += leafNodes[i].bounding_box;
	}
	if (idx == 0)
	{
		bvh->root = &internalNodes[idx];
		bvh->LeavesPtr = leafNodes;
		bvh->numObjects = numObjects;
	}
}

__global__ void setID(LeafNode* leafNodes, unsigned int* sortedObjectIDs, size_t numObjects, float* _vtxs, tinyobj::index_t* _tris)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numObjects)
		return;
	leafNodes[i].objectID = sortedObjectIDs[i];
	leafNodes[i].isleaf = true;
	tinyobj::index_t* tri1 = &_tris[sortedObjectIDs[i] * 3];
	tinyobj::index_t* tri2 = &_tris[sortedObjectIDs[i] * 3 + 1];
	tinyobj::index_t* tri3 = &_tris[sortedObjectIDs[i] * 3 + 2];
	leafNodes[i].bounding_box.set(Vec3(_vtxs[tri1->vertex_index * 3], _vtxs[tri1->vertex_index * 3 + 1], _vtxs[tri1->vertex_index * 3 + 2]));
	leafNodes[i].bounding_box += Vec3(_vtxs[tri2->vertex_index * 3], _vtxs[tri2->vertex_index * 3 + 1], _vtxs[tri2->vertex_index * 3 + 2]);
	leafNodes[i].bounding_box += Vec3(_vtxs[tri3->vertex_index * 3], _vtxs[tri3->vertex_index * 3 + 1], _vtxs[tri3->vertex_index * 3 + 2]);
	//printf("%d %lf %lf %lf\n", i, leafNodes[i].bounding_box._min[0], leafNodes[i].bounding_box._min[1], leafNodes[i].bounding_box._min[2]);
}

NodePtr generateHierarchy(LeafNode* leafNodes, InternalNode* internalNodes, unsigned int* sortedMortonCodes,
                          unsigned int* sortedObjectIDs, size_t numObjects, BVH* bvh,
                          float* _vtxs,tinyobj::index_t* _tris)
{
	setID << <numObjects / 256 + 1, 256 >> > (leafNodes, sortedObjectIDs, numObjects, _vtxs, _tris);
	createNode << <(numObjects - 1) / 256 + 1, 256 >> > (leafNodes, internalNodes, sortedMortonCodes, sortedObjectIDs,
	                                                     numObjects, bvh);
	return &internalNodes[0];
}

NodePtr generateHierarchy_long(LeafNode* leafNodes, InternalNode* internalNodes, unsigned long long* sortedMortonCodes,
	unsigned int* sortedObjectIDs, size_t numObjects, BVH* bvh,
	float* _vtxs, tinyobj::index_t* _tris)
{
	setID << <numObjects / 256 + 1, 256 >> > (leafNodes, sortedObjectIDs, numObjects, _vtxs, _tris);
	createNode_long << <(numObjects - 1) / 256 + 1, 256 >> > (leafNodes, internalNodes, sortedMortonCodes, sortedObjectIDs,
		numObjects, bvh);
	return &internalNodes[0];
}

void Debug2(
	unsigned long long* sortedMortonCodes, int numtri,
	unsigned int* sortedObjectIDs)
{
	unsigned long long* sortedMortonCode = new unsigned long long[numtri];
	unsigned int* sortedObjectID = new unsigned int[numtri];
	//int idx = threadIdx.x + blockIdx.x * blockDim.x;
	HANDLE_ERROR(cudaMemcpy(sortedMortonCode, sortedMortonCodes, sizeof(unsigned long long) * numtri, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(sortedObjectID, sortedObjectIDs, sizeof(unsigned int) * numtri, cudaMemcpyDeviceToHost));

	for (int idx=0;idx<numtri;idx++)
	{
		printf("%d %llu %u\t", idx, sortedMortonCode[idx], sortedObjectID[idx]);
	}
	//if (idx >= numtri)
	//	return;
	delete[] sortedMortonCode;
	delete[] sortedObjectID;
}

GPU_DataSet::GPU_DataSet(const Model& model, int h, int w)
{
	height = h; width = w;
	numtri = model.shapes[0][0].mesh.indices.size() / 3, numvtx = model.attrib[0].vertices.size() / 3;

	HANDLE_ERROR(cudaMalloc((void**)&dev_ids, sizeof(tinyobj::index_t) * 3 * numtri));
	HANDLE_ERROR(cudaMemcpy(dev_ids, model.shapes[0][0].mesh.indices.data(), sizeof(tinyobj::index_t) * 3 * numtri, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&dev_vtxs, sizeof(float) * 3 * numvtx));
	HANDLE_ERROR(cudaMemcpy(dev_vtxs, model.attrib[0].vertices.data(), sizeof(float) * 3 * numvtx, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&dev_nmls, sizeof(float) * 3 * numvtx));
	HANDLE_ERROR(cudaMemcpy(dev_nmls, model.attrib[0].normals.data(), sizeof(float) * 3 * numvtx, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&dev_cols, sizeof(float) * 3 * numvtx));
	HANDLE_ERROR(cudaMemcpy(dev_cols, model.attrib[0].colors.data(), sizeof(float) * 3 * numvtx, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&dev_texs, sizeof(float) * 2 * numvtx));
	HANDLE_ERROR(cudaMemcpy(dev_texs, model.attrib[0].texcoords.data(), sizeof(float) * 2 * numvtx, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&dev_matids, sizeof(int) * numtri));
	HANDLE_ERROR(cudaMemcpy(dev_matids, model.shapes[0][0].mesh.material_ids.data(), sizeof(int) * model.shapes[0][0].mesh.material_ids.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&materials, sizeof(Material) * model.material->size()));
	HANDLE_ERROR(cudaMemcpy(materials, model.materials.data(), sizeof(Material) * model.materials.size(), cudaMemcpyHostToDevice));

	ligsize = model.lights.size();
	HANDLE_ERROR(cudaMalloc((void**)&dev_light, sizeof(int) * ligsize));
	HANDLE_ERROR(cudaMemcpy(dev_light, model.lights.data(), sizeof(int) * ligsize, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&dev_morton_code_array, sizeof(unsigned long long) * numtri));
	HANDLE_ERROR(cudaMalloc((void**)&dev_index_array, sizeof(unsigned int) * numtri));
	HANDLE_ERROR(cudaMalloc((void**)&leafNodes, sizeof(LeafNode) * numtri));
	HANDLE_ERROR(cudaMalloc((void**)&internalNodes, sizeof(InternalNode) * (numtri - 1)));
	HANDLE_ERROR(cudaMalloc((void**)&bvh, sizeof(BVH)));

	dim3 blockSize(numtri / 256 + 1);
	dim3 threadSize(256);

	/*makeMorton << <blockSize, threadSize >> > (numtri, dev_ids, dev_vtxs, dev_morton_code_array,
	                                           dev_index_array, model.box._max[0] - model.box._min[0], model.box._max[1] - model.box._min[1], model.box._max[2] - model.box._min[2],
	                                           model.box._min[0], model.box._min[1], model.box._min[2]);*/

	makeMorton_long << <blockSize, threadSize >> > (numtri, dev_ids, dev_vtxs, dev_morton_code_array,
		dev_index_array, model.box._max[0] - model.box._min[0], model.box._max[1] - model.box._min[1], model.box._max[2] - model.box._min[2],
		model.box._min[0], model.box._min[1], model.box._min[2]);

	
	thrust::sort_by_key(thrust::device, dev_morton_code_array, dev_morton_code_array + numtri,
	                    dev_index_array);
	
	Node* root = generateHierarchy_long(leafNodes, internalNodes, dev_morton_code_array, dev_index_array, numtri, bvh, dev_vtxs, dev_ids);

	dim3 blockSize2(width / 16 + 1, height / 16 + 1);
	dim3 threadSize2(16, 16);

	HANDLE_ERROR(cudaMalloc((void**)&dev_states, sizeof(curandState) * width * height));


	HANDLE_ERROR(cudaMalloc((void**)&dev_image, sizeof(DoubleColor) * height * width));
	image = new DoubleColor[height * width];
	
	cudaFree(dev_morton_code_array);
	cudaFree(dev_index_array);

	load_textures(model);
}

GPULINE bool GPU_DataSet::nobvh_hit(double t_min, double t_max, const Ray& r, hit_record& record)const
{
	Vec3 normal, point, center;
	double t = t_max, mint = t_max;
	float area = 0;
	bool is_pre;
	for (int id = 0; id < numtri; id++)
	{
		if (RayTriangleIntersect(r, t,area, is_pre, dev_ids[id * 3], dev_ids[id * 3 + 1], dev_ids[id * 3 + 2], normal, point, center))
		{
			if (t > t_min && t < t_max && t < mint)
			{
				mint = t;
				record.t = mint;
				record.point = point;
				record.normal = unit_vector(normal);
				record.u = center.x * dev_texs[dev_ids[id * 3].texcoord_index * 2] + center.y * dev_texs[dev_ids[id * 3 + 1].texcoord_index * 2] + center.z * dev_texs[dev_ids[id * 3 + 2].texcoord_index * 2];
				record.v = center.x * dev_texs[dev_ids[id * 3].texcoord_index * 2 + 1] + center.y * dev_texs[dev_ids[id * 3 + 1].texcoord_index * 2 + 1] + center.z * dev_texs[dev_ids[id * 3 + 2].texcoord_index * 2 + 1];
				record.color = center.x * Vec3(dev_cols[dev_ids[id * 3].vertex_index * 3], dev_cols[dev_ids[id * 3].vertex_index * 3 + 1], dev_cols[dev_ids[id * 3].vertex_index * 3 + 2])
					+ center.y * Vec3(dev_cols[dev_ids[id * 3 + 1].vertex_index * 3], dev_cols[dev_ids[id * 3 + 1].vertex_index * 3 + 1], dev_cols[dev_ids[id * 3 + 1].vertex_index * 3 + 2])
					+ center.z * Vec3(dev_cols[dev_ids[id * 3 + 2].vertex_index * 3], dev_cols[dev_ids[id * 3 + 2].vertex_index * 3 + 1], dev_cols[dev_ids[id * 3 + 2].vertex_index * 3 + 2]);

				record.is_preface = is_pre;
				record.tri_id = id;
				record.mat_ptr = &materials[dev_matids[id]];
			}
		}
	}
	if (mint > t_min && mint < t_max)
		return record.is_preface;
	else
		return false;
}

GPULINE bool GPU_DataSet::hit(double t_min, double& t_max, const Ray& r, hit_record& record, int id)const
{
	double t = 0;
	Vec3 normal, point, center;
	float area = 0;
	bool is_pre;
	if (RayTriangleIntersect(r, t, area, is_pre, dev_ids[id * 3], dev_ids[id * 3 + 1], dev_ids[id * 3 + 2], normal, point, center))
	{
		if (t > t_min && t <= t_max)
		{
			record.t = t;
			record.point = point;
			record.normal = unit_vector(normal);
			record.u = center.x * dev_texs[dev_ids[id * 3].texcoord_index * 2] + center.y * dev_texs[dev_ids[id * 3 + 1].texcoord_index * 2] + center.z * dev_texs[dev_ids[id * 3 + 2].texcoord_index * 2];
			record.v = center.x * dev_texs[dev_ids[id * 3].texcoord_index * 2 + 1] + center.y * dev_texs[dev_ids[id * 3 + 1].texcoord_index * 2 + 1] + center.z * dev_texs[dev_ids[id * 3 + 2].texcoord_index * 2 + 1];
			record.color = center.x * Vec3(dev_cols[dev_ids[id * 3].vertex_index * 3], dev_cols[dev_ids[id * 3].vertex_index * 3 + 1], dev_cols[dev_ids[id * 3].vertex_index * 3 + 2])
				+ center.y * Vec3(dev_cols[dev_ids[id * 3 + 1].vertex_index * 3], dev_cols[dev_ids[id * 3 + 1].vertex_index * 3 + 1], dev_cols[dev_ids[id * 3 + 1].vertex_index * 3 + 2])
				+ center.z * Vec3(dev_cols[dev_ids[id * 3 + 2].vertex_index * 3], dev_cols[dev_ids[id * 3 + 2].vertex_index * 3 + 1], dev_cols[dev_ids[id * 3 + 2].vertex_index * 3 + 2]);

			record.mat_ptr = &materials[dev_matids[id]];
			record.mat_id = dev_matids[id];
			record.tri_id = id;
			record.is_preface = is_pre;
			record.area = area;
			t_max = t;
			return is_pre;
		}
	}
	return false;
}

DoubleColor GPU_DataSet::light_color(hit_record& rec,const Ray& ray) const
{
	Vec3 light(0, 0, 0);
	Vec3 reflected = reflect(unit_vector(ray.GetDirection()), unit_vector(rec.normal));
	for (int i = 0; i < ligsize; i++)
	{
		int id = dev_light[i];
		hit_record record;
		double u, v;
		double pdf_value= 0.0001;
		u = gpu_random_double(dev_states, width) * 1.0;
		v = gpu_random_double(dev_states, width) * (1.0 - u);

		Vec3 random_point(
			dev_vtxs[dev_ids[id * 3].vertex_index * 3] * (1 - u - v) + dev_vtxs[dev_ids[id * 3 + 1].vertex_index * 3] * u +
			dev_vtxs[dev_ids[id * 3 + 2].vertex_index * 3] * v,
			dev_vtxs[dev_ids[id * 3].vertex_index * 3 + 1] * (1 - u - v) + dev_vtxs[dev_ids[id * 3 + 1].vertex_index * 3 + 1
			] * u + dev_vtxs[dev_ids[id * 3 + 1].vertex_index * 3 + 1] * v,
			dev_vtxs[dev_ids[id * 3].vertex_index * 3 + 2] * (1 - u - v) + dev_vtxs[dev_ids[id * 3 + 1].vertex_index * 3 + 2
			] * u + dev_vtxs[dev_ids[id * 3 + 2].vertex_index * 3 + 2] * v);
		Ray r(rec.point, random_point - rec.point);
		if (!bvh_hit(EPSILON, 1.000, r, record))
		{
			continue;
		}
		auto area = record.area;
		auto distance_squre = record.t * record.t * r.GetDirection().square();
		auto cosine = fabs(dot(r.GetDirection(), record.normal) / r.GetDirection().length());
		pdf_value= distance_squre / (cosine * area);
		float cosine2;
		cosine2 = dot(rec.normal, unit_vector(r.GetDirection()));
		double pdf_value2= cosine2 < 0 ? 0 : cosine2 / PI;
		/*
		if(rec.mat_ptr->Ns>1)
		printf("%f %f %f\t", dot(unit_vector(r.GetDirection()), reflected),powf(
			dot(unit_vector(r.GetDirection()), reflected), rec.mat_ptr->Ns) * pdf_value2 / pdf_value, pdf_value2 / pdf_value);*/
		light += record.mat_ptr->Le.ScalarProduct(rec.mat_ptr->Kd * pdf_value2 / pdf_value + rec.mat_ptr->Ks * powf(
			dot(unit_vector(r.GetDirection()), reflected), rec.mat_ptr->Ns)* pdf_value2 / pdf_value);
	}
	return light;
}

__device__ Vec3 GPU_DataSet::random(curandState* curand_states,const Vec3& o) const
{
	int id = dev_light[(int)gpu_random_double_range(curand_states, width, 0, ligsize)];
	double u, v;
	u = gpu_random_double(curand_states, width);
	v = gpu_random_double(curand_states, width);

	Vec3 random_point(
		dev_vtxs[dev_ids[id * 3].vertex_index * 3] * (1 - u - v) + dev_vtxs[dev_ids[id * 3 + 1].vertex_index * 3] * u +
		dev_vtxs[dev_ids[id * 3 + 2].vertex_index * 3] * v,
		dev_vtxs[dev_ids[id * 3].vertex_index * 3 + 1] * (1 - u - v) + dev_vtxs[dev_ids[id * 3 + 1].vertex_index * 3 + 1
		] * u + dev_vtxs[dev_ids[id * 3 + 1].vertex_index * 3 + 1] * v,
		dev_vtxs[dev_ids[id * 3].vertex_index * 3 + 2] * (1 - u - v) + dev_vtxs[dev_ids[id * 3 + 1].vertex_index * 3 + 2
		] * u + dev_vtxs[dev_ids[id * 3 + 2].vertex_index * 3 + 2] * v);
	return random_point - o;
}

__device__ double GPU_DataSet::pdf_value(const Ray& r, float infinity, bool& needStop, hit_record& record) const
{
	if (!bvh_hit(EPSILON, infinity, r, record))
	{
		needStop = true;
		return 0.0001;
	}
	auto area = record.area;
	auto distance_squre = record.t * record.t * r.GetDirection().square();
	auto cosine = fabs(dot(r.GetDirection(), record.normal) / r.GetDirection().length());
	return distance_squre / (cosine * area);
}

void GPU_DataSet::deleteMySelf()
{
	/*for (int i = 0; i < matsize; i++)
	{
		Material* temp;

		HANDLE_ERROR(cudaMemcpy(&temp, &materials[i], sizeof(Material*), cudaMemcpyDeviceToHost));
		cudaFree(temp);

	}*/
	for (int i = 0; i < texsize; i++)
	{
		cudaDestroyTextureObject(texid[i]);
	}if (scene_id!= 0)
	{
		cudaDestroyTextureObject(scene_id);
	}
	delete[] texid;
	cudaFree(dev_texid);
	cudaFree(dev_vtxs);
	cudaFree(dev_ids);

	cudaFree(dev_nmls);
	cudaFree(dev_cols);
	cudaFree(dev_texs);
	cudaFree(dev_matids);
	cudaFree(materials);
	cudaFree(leafNodes);
	cudaFree(internalNodes);
	cudaFree(bvh);
	delete[] image;
	cudaFree(dev_image);
	cudaFree(dev_states);
	cudaFree(dev_light);
}

__device__ double PDF::value(const Ray& r, float infinity, bool& needTraverse,hit_record& record) const
{
	switch (type)
	{
	case 1:
		return ptr->pdf_value(r, infinity, needTraverse, record);
	case 2:
		return pptr[0]->value(r.GetDirection()) * weight + pptr[1]->ptr->pdf_value(r, infinity, needTraverse, record) *
			(1 - weight);
	default:
		return 0;
		break;
	}
}

__device__ double PDF::value(const Vec3& direction)const
{
	float cosine;
	switch (type)
	{
	case 0:
		cosine = dot(uvw.w(), unit_vector(direction));
		return cosine < 0 ? 0 : cosine / PI;
	default:
		return 0;
		break;
	}
}

__device__ Vec3 PDF::generate(curandState* curand_states, int width, const Vec3& o) const
{
	switch (type)
	{
	case 0:
		return uvw.local(gpu_random_cosine_n_direction(curand_states, width, Ns));
	case 1:
		return  ptr->random(curand_states, o);
	case 2:
		if (pptr[1]->ptr->ligsize == 0 || gpu_random_double(curand_states, width) < weight)
			return pptr[0]->uvw.local(gpu_random_cosine_n_direction(curand_states, width, Ns));
		else
			return pptr[1]->ptr->random(curand_states, o);
		
	default:
		return Vec3(0, 0, 0);
		break;
	}
}

__device__ Vec3 PDF::generate(curandState* curand_states, int width) const
{
	switch (type)
	{
	case 0:
		return uvw.local(gpu_random_cosine_n_direction(curand_states, width, Ns));
	default:
		return Vec3(0, 0, 0);
		break;
	}
}

GPULINE bool GPU_DataSet::RayTriangleIntersect(
	const Ray& r, double& t, float& area, bool& is_preface,
	const tinyobj::index_t& n0, const tinyobj::index_t& n1, const tinyobj::index_t& n2,
	Vec3& normal, Vec3& Point, Vec3& center)const
{
	Vec3 v0(dev_vtxs[n0.vertex_index * 3], dev_vtxs[n0.vertex_index * 3 + 1], dev_vtxs[n0.vertex_index * 3 + 2]);
	Vec3 v1(dev_vtxs[n1.vertex_index * 3], dev_vtxs[n1.vertex_index * 3 + 1], dev_vtxs[n1.vertex_index * 3 + 2]);
	Vec3 v2(dev_vtxs[n2.vertex_index * 3], dev_vtxs[n2.vertex_index * 3 + 1], dev_vtxs[n2.vertex_index * 3 + 2]);
	Vec3 no0(dev_nmls[n0.normal_index * 3], dev_nmls[n0.normal_index * 3 + 1], dev_nmls[n0.normal_index * 3 + 2]);
	Vec3 no1(dev_nmls[n1.normal_index * 3], dev_nmls[n1.normal_index * 3 + 1], dev_nmls[n1.normal_index * 3 + 2]);
	Vec3 no2(dev_nmls[n2.normal_index * 3], dev_nmls[n2.normal_index * 3 + 1], dev_nmls[n2.normal_index * 3 + 2]);

	bool isIn = false;
	Vec3 E1 = v1 - v0;
	Vec3 E2 = v2 - v0;
	Vec3 S = r.origin - v0;
	Vec3 S1 = cross(r.direction, E2);
	Vec3 S2 = cross(S, E1);
	area = cross(E1, E2).length() / 2;
	float coeff = 1.0 / dot(S1, E1);
	float tt = coeff * dot(S2, E2);
	float b1 = coeff * dot(S1, S);
	float b2 = coeff * dot(S2, r.direction);
	if (t >= 0 && b1 >= 0 && b2 >= 0 && (1 - b1 - b2) >= 0)
	{
		isIn = true;
		t = tt;
		center.x = 1 - b1 - b2;
		center.y = b1;
		center.z = b2;
		Point = r.origin + t * r.direction;
		normal = center.x * unit_vector(no0)+ center.y * unit_vector(no1) + center.z * unit_vector(no2);
		//normal = center.x * no0 + center.y * no1 + center.z * no2;
		is_preface = dot(normal, r.direction) < 0;
	}
	return isIn;
}


__device__ bool GPU_DataSet::bvh_hit(double t_min, double t_max, const Ray& r, hit_record& record)const 
{
	bool is_hit = false;
	NodePtr stack[64];
	NodePtr* stackPtr = stack;
	*stackPtr++ = NULL;
	record.is_preface = true;
	NodePtr node = bvh->getRoot();
	do
	{
		NodePtr childL = bvh->getLeftChild(node);
		NodePtr childR = bvh->getRightChild(node);
		bool overlapL = childL->bounding_box.hit(t_min, t_max, r);
		bool overlapR = childR->bounding_box.hit(t_min, t_max, r);

		if (overlapL && bvh->isLeaf(childL))
		{
			hit(t_min, t_max, r, record, childL->objectID) ? is_hit = true: 0;
			if (childL->objectID == 187986)
				record.hit_number++;
		}

		if (overlapR && bvh->isLeaf(childR))
		{
			hit(t_min, t_max, r, record, childR->objectID) ? is_hit = true: 0;
			if (childR->objectID == 187986)
				record.hit_number++;
		}

		bool traverseL = (overlapL && !bvh->isLeaf(childL));
		bool traverseR = (overlapR && !bvh->isLeaf(childR));

		if (!traverseL && !traverseR)
			node = *--stackPtr;
		else
		{
			node = (traverseL) ? childL : childR;
			if (traverseL && traverseR)
				*stackPtr++ = childR;
		}
	} while (node != NULL);
	return is_hit;//&& record.is_preface;
}

void GPU_DataSet::load_textures(const Model& model)
{
	texsize = model.texture_nemes.size();
	texid = new cudaTextureObject_t[texsize];
	for (auto i = 0; i < texsize; i++)
	{
		int width, height, depth;
		const auto tex_data = stbi_load(model.texture_nemes[i].c_str(), &width, &height, &depth, 0);
		if(tex_data==NULL)
			continue;
		const auto size = width * height * depth;
		float* h_data = new float[size];

		for (unsigned int layer = 0; layer < 3; layer++)
			for (auto i = 0; i < static_cast<int>(width * height); i++)h_data[layer * width * height + i] = tex_data[i *
				3 + layer] / 255.0;
		
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		
		cudaArray* d_cuArr;
		cudaMalloc3DArray(&d_cuArr, &channelDesc, make_cudaExtent(width, height, 3), cudaArrayLayered);

		cudaMemcpy3DParms myparms = { 0 };
		myparms.srcPos = make_cudaPos(0, 0, 0);
		myparms.dstPos = make_cudaPos(0, 0, 0);
		myparms.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(float), width, height);
		myparms.dstArray = d_cuArr;
		myparms.extent = make_cudaExtent(width, height, 3);
		myparms.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3D(&myparms);


		cudaResourceDesc    texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = d_cuArr;
		cudaTextureDesc     texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeWrap;
		texDescr.addressMode[1] = cudaAddressModeWrap;
		texDescr.addressMode[2] = cudaAddressModeWrap;
		texDescr.readMode = cudaReadModeElementType;
		texDescr.normalizedCoords = true;
		cudaCreateTextureObject(&texid[i], &texRes, &texDescr, NULL);

		delete[]h_data;
		delete[]tex_data;
	}
	HANDLE_ERROR(cudaMalloc((void**)&dev_texid, sizeof(cudaTextureObject_t) * texsize));
	HANDLE_ERROR(cudaMemcpy(dev_texid, texid, sizeof(cudaTextureObject_t) * texsize, cudaMemcpyHostToDevice));
	if(model.scene_path!="")
	{
		int width, height, depth;
		const auto tex_data = stbi_load(model.scene_path.c_str(), &width, &height, &depth, 0);
		if (tex_data == NULL)
			return;
		const auto size = width * height * depth;
		float* h_data = new float[size];

		for (unsigned int layer = 0; layer < 3; layer++)
			for (auto i = 0; i < static_cast<int>(width * height); i++)h_data[layer * width * height + i] = tex_data[i *
				3 + layer] / 255.0 * tex_data[i * 3 + layer] / 255.0;

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		cudaArray* d_cuArr;
		cudaMalloc3DArray(&d_cuArr, &channelDesc, make_cudaExtent(width, height, 3), cudaArrayLayered);

		cudaMemcpy3DParms myparms = { 0 };
		myparms.srcPos = make_cudaPos(0, 0, 0);
		myparms.dstPos = make_cudaPos(0, 0, 0);
		myparms.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(float), width, height);
		myparms.dstArray = d_cuArr;
		myparms.extent = make_cudaExtent(width, height, 3);
		myparms.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3D(&myparms);


		cudaResourceDesc    texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = d_cuArr;
		cudaTextureDesc     texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeWrap;
		texDescr.addressMode[1] = cudaAddressModeWrap;
		texDescr.addressMode[2] = cudaAddressModeWrap;
		texDescr.readMode = cudaReadModeElementType;
		texDescr.normalizedCoords = true;
		cudaCreateTextureObject(&scene_id, &texRes, &texDescr, NULL);
		delete[]h_data;
		delete[]tex_data;
	}
}

__device__ DoubleColor ray_color(const Ray& r, const DoubleColor& background, GPU_DataSet& model, int depth,
                                 double infinity)
{
	hit_record record, record2;
	DoubleColor color(0.0, 0.0, 0.0);
	int max_depth = 0;
	struct ColorStack
	{
		bool plus;
		DoubleColor emit;
		DoubleColor attenuation;
		bool is_spe=false;
	};
	ColorStack cs_stack[20];
	Ray scattered = r;
	bool needStop = false;
	for (int i = 0; i < depth; i++)
	{
		max_depth++;
		if (!model.bvh_hit(0.0001, infinity, scattered, record))
		//if ((i == 0 && !model.bvh_hit(0.0001, infinity, scattered, record)) || needStop)
		{
			if (model.scene_id != 0)
			{
				double u, v;
				get_sphere_uv(unit_vector(scattered.GetDirection()), u, v);
				/*
				 * 1: 1.3 - u, -v
				 * 2: 0.8 - u, -v 
				 * 3: 0.85 - u, -v
				 * 4: 0.16 - u, -v
				 */
				cs_stack[i].emit = Vec3(
					tex2DLayered<float>(model.scene_id, 0.85 - u, -v, 0),
					tex2DLayered<float>(model.scene_id, 0.85 - u, -v, 1),
					tex2DLayered<float>(model.scene_id, 0.85 - u, -v, 2));
			}else
				cs_stack[i].emit = background;
			cs_stack[i].attenuation = DoubleColor(0.0, 0.0, 0.0);
			cs_stack[i].plus = true;
			break;
		}
		double pdf_value = 1;
		scatter_record srecord;
		DoubleColor emitted = record.mat_ptr->emitted(scattered, record, record.u, record.v, record.point);

		if (!record.mat_ptr->scatter(scattered, record, srecord,model.dev_states,model.width))
		{
			cs_stack[i].attenuation = DoubleColor(0.0, 0.0, 0.0);
			cs_stack[i].emit = emitted;
			cs_stack[i].plus = true;
			break;
		}
		float w = 0.0;
		
		if (srecord.is_specular)
		{
			//cs_stack[i].emit = srecord.attenuation.ScalarProduct((model.light_color(record, scattered)));
			scattered = Ray(record.point, srecord.pdf_ptr.generate(model.dev_states, model.width),r.GetTime());
			//scattered = srecord.specular_ray;
			pdf_value = srecord.pdf_ptr.value(scattered.GetDirection());
			cs_stack[i].attenuation = srecord.attenuation.ScalarProduct(record.mat_ptr->Ks);// *powf(dot(srecord.specular_ray.GetDirection(), scattered.GetDirection()), record.mat_ptr->Ns); //  .ScalarProduct(record.mat_ptr->Ks + record.mat_ptr->Kd * record.mat_ptr->scattering_pdf(r, record, scattered) / pdf_value);
			/* srecord.attenuation.ScalarProduct(record.mat_ptr->Kd * record.mat_ptr->scattering_pdf(r, record, scattered) / pdf_value +
							record.mat_ptr->Ks * powf(dot(srecord.specular_ray.GetDirection(), scattered.GetDirection()), record.mat_ptr->Ns)); */
			cs_stack[i].plus = false;

			/*pdf_value = srecord.pdf_ptr.value(scattered, infinity, needStop, record2);
			record = record2;*/
			//Vec3 reflected = reflect(unit_vector(r.GetDirection()), record.normal);

			/*if (!model.bvh_hit(EPSILON, infinity, scattered, record))
				needStop = true;*/
			continue;
		}
		
		//PDF light_ptr(&model, record.point);
		//PDF p(&srecord.pdf_ptr, &light_ptr, w);

		//scattered = Ray( record.point, p.generate(model.dev_states, model.width, record.point), r.GetTime());
		//pdf_value = p.value(scattered, infinity, needStop, record2);

		//cs_stack[i].emit = srecord.attenuation.ScalarProduct(model.light_color(record, scattered));
		scattered = Ray( record.point, srecord.pdf_ptr.generate(model.dev_states, model.width), r.GetTime());
		pdf_value = srecord.pdf_ptr.value(scattered.GetDirection());

		cs_stack[i].attenuation = srecord.attenuation.ScalarProduct(
			record.mat_ptr->Kd * record.mat_ptr->scattering_pdf(r, record, scattered) / pdf_value);
		cs_stack[i].plus = false;
		//record = record2;
	}
	
	for (int i = max_depth - 1; i >= 0; i--)
	{
		if(cs_stack[i].plus)
		{
			color = cs_stack[i].emit + cs_stack[i].attenuation.ScalarProduct(color);
		}else
		{
			color = cs_stack[i].attenuation.ScalarProduct(color);
		}
	}
	/*unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
	if(idy<100)
		printf("%f %f %f\n", color.x, color.y, color.z);*/
	return color;
}

__global__ void GPU_Tracer(int samples_per_pixel, DoubleColor* image,DoubleColor background, 
                           GPU_DataSet model, int depth ,Camera camera, double infinity)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idx>= model.width ||idy >= model.height)
		return;
	DoubleColor pixel_color(0, 0, 0);
	for (int k = 0; k < samples_per_pixel; k++)
	{
		double u = (double(idy) + gpu_random_double(model.dev_states, model.width)) / (model.height - 1);
		double v = (double(idx) + gpu_random_double(model.dev_states, model.width)) / (model.width - 1);
		
		Ray	r = camera.Get_Ray(u, v);
		pixel_color += ray_color(r, background, model, depth,infinity);
	}
	image[idy * model.width + idx] = pixel_color / samples_per_pixel;
}

__global__ void gpu_set_random(curandState* curand_states, int width, int height, long clock_for_rand)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx + idy * width >= width * height)
		return;
	curand_init(clock_for_rand + idx + idy * width, 0, 0, &curand_states[idx + idy * width]);
}


GPU_DataSet CreateGPUModel(const Model& model,int height,int width)
{
	GPU_DataSet gmodel(model, height, width);
	dim3 blockSize(gmodel.width / 16 + 1, gmodel.height / 16 + 1);
	dim3 threadSize(16, 16);
	long clock_for_rand = clock();

	gpu_set_random << <blockSize, threadSize >> > (gmodel.dev_states, gmodel.width, gmodel.height, clock_for_rand);
	return gmodel;
}

__global__ void Debug(
	LeafNode* leafNodes,int numtri,
	InternalNode* internalNodes)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx>=numtri-1)
		return;
	if(internalNodes[idx].chiA == -1 && internalNodes[idx].childA->objectID == 187986)
		printf("%d l\n", idx);
	if(internalNodes[idx].chiB == -1 && internalNodes[idx].childB->objectID == 187986)
		printf("%d r\n", idx);
	//if(internalNodes[idx].)
	/*printf("%d %f %f %f %f %f %f\n", internalNodes[idx].objectID, internalNodes[idx].bounding_box._min.x,
		internalNodes[idx].bounding_box._min.y, internalNodes[idx].bounding_box._min.z
	       , internalNodes[idx].bounding_box._max.x, internalNodes[idx].bounding_box._max.y,
		internalNodes[idx].bounding_box._max.z);*/
}

void RayTracing(const GPU_DataSet& model, int depth, SDLImage& screen,int samples_per_pixel,
                DoubleColor background)
{
	cudaEvent_t start, end;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	HANDLE_ERROR(cudaEventSynchronize(start));

	//dim3 blockSize2(model.numtri / 256 + 1);
	//dim3 threadSize2(256);
	//Debug << <blockSize2, threadSize2 >> > (model.leafNodes, model.numtri, model.internalNodes);
	
	dim3 blockSize(model.width / 16 + 1, model.height / 16 + 1);
	dim3 threadSize(16, 16);
	
	
	GPU_Tracer << <blockSize, threadSize >> > (samples_per_pixel, model.dev_image,
	                                           background, model, depth, screen.camera,infinity);

	HANDLE_ERROR(
		cudaMemcpy(model.image, model.dev_image, sizeof(DoubleColor) * model.height * model.width,cudaMemcpyDeviceToHost
		));

	for (int i = 0; i < model.height; i++)
	{
		for (int j = 0; j < model.width; j++)
		{
			screen.set(j, i, model.image[i * model.width + j].getx(), model.image[i * model.width + j].gety(),
			           model.image[i * model.width + j].getz(),samples_per_pixel);
		}
	}

	HANDLE_ERROR(cudaEventCreate(&end));
	HANDLE_ERROR(cudaEventRecord(end, 0));
	HANDLE_ERROR(cudaEventSynchronize(end));

	float Time_Elapse;
	HANDLE_ERROR(cudaEventElapsedTime(&Time_Elapse, start, end));

	printf("time= %lf s FPS= %lf\n", Time_Elapse/1000, 1000 / Time_Elapse);
	
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(end));
}