#pragma once
#include <memory>
#include "image.h"
#include "ray.h"
#include "common.h"
//#include "texture.h"
#include "pdf.h"

extern class Material;

#if USEGPU==true

struct hit_record
{
	double t;
	Vec3 normal;
	Vec3 point;
	double u, v;
	Material* mat_ptr;
	DoubleColor color;
	int mat_id;
	int tri_id;
	float area;
	bool is_preface;
	int hit_number = 0;
};
struct scatter_record
{
	Ray	specular_ray;
	bool is_specular;
	DoubleColor attenuation;
	PDF pdf_ptr;
};

class Material
{
public:
	GPULINE Material() {}
	GPULINE Material(const Vec3& kd, const Vec3& ks, double ns,int _type,int texid) : Kd(kd), Ks(ks), Ns(ns), type(_type), textureid(texid)
	{
		
	}
	GPULINE Material(const Vec3& kd, const Vec3& ks, double ns, const Vec3& le, int _type) : Kd(kd), Ks(ks), Ns(ns), Le(le), type(_type)
	{

	}
	__device__ DoubleColor emitted(const Ray& r, const hit_record& rec, double u, double v, const Vec3& p)const
	{
		switch (type)
		{
			case 0:
				return DoubleColor(0, 0, 0);
			case 1:
				return rec.color.ScalarProduct(Le);
			case 2:
				return DoubleColor(0, 0, 0);
			default:
				return DoubleColor(0, 0, 0);
		}
	}
	
	__device__ bool scatter(const Ray& r, const hit_record& rec, scatter_record& srec, curandState* curand_states, int width)const
	{
		switch (type)
		{
		case 0:
			srec.is_specular = false;
			if (textureid != -1)
				srec.attenuation = rec.color.ScalarProduct(Vec3(
					tex2DLayered<float>(textureid, rec.u, 1 - rec.v, 0),
					tex2DLayered<float>(textureid, rec.u, 1 - rec.v, 1),
					tex2DLayered<float>(textureid, rec.u, 1 - rec.v, 2)));
			else
				srec.attenuation = rec.color;

			srec.pdf_ptr = PDF(rec.normal, Ns);
			return true;
		case 1:
			return false;
		case 2:
			{
			Vec3 reflected = reflect(unit_vector(r.GetDirection()), unit_vector(rec.normal));

			float Is = 0.27 * Ks.x + 0.67 * Ks.y + 0.06 * Ks.z;
			float Id = 0.27 * Kd.x + 0.67 * Kd.y + 0.06 * Kd.z;

			if(gpu_random_double(curand_states, width)<(Is/(Is+Id)))
			{
				srec.pdf_ptr = PDF(reflected, Ns);
				srec.is_specular = true;
			}else
			{
				srec.pdf_ptr = PDF(rec.normal, 1);
				srec.is_specular = false;
			}
			//srec.pdf_ptr = PDF(reflected, Ns);
				
			if (textureid != -1)
				srec.attenuation = rec.color.ScalarProduct(Vec3(
					tex2DLayered<float>(textureid, rec.u, 1 - rec.v, 0),
					tex2DLayered<float>(textureid, rec.u, 1 - rec.v, 1),
					tex2DLayered<float>(textureid, rec.u, 1 - rec.v, 2)));
			else
				srec.attenuation = rec.color;
				
			srec.specular_ray = Ray(rec.point, reflected);
			return true;
			}
		default:
			return true;
		}
	}
	
	GPULINE double scattering_pdf(const Ray& r_in, const hit_record& rec, Ray& scatter)const
	{
		switch (type)
		{
		case 0:
			{
			float cosine;
			cosine = dot(rec.normal, unit_vector(scatter.GetDirection()));
			return cosine < 0 ? 0 : cosine / PI;
			}
		case 1:
			return 0;
			break;
		case 2: 
		{
			float cosine;
			cosine = dot(rec.normal, unit_vector(scatter.GetDirection()));
			return cosine < 0 ? 0 : cosine / PI;
		}
			break;
		default:
			return 0;
		}
	}
	
	~Material()
	{

	}
public:
	int textureid;
	Vec3 Kd, Ks;
	double Ns;
	Vec3 Le;
	int type;

	/*
	 * type
	 * 0:漫反射材质
	 * 1:光照
	 * 2:glossy
	 * 3.镜面
	 */
};

#else

struct hit_record
{
	double t;
	Vec3 normal;
	Vec3 point;
	double u, v;
	shared_ptr<Material> mat_ptr;
	bool front_face;
	DoubleColor color;
};
struct scatter_record
{
	Ray	specular_ray;
	bool is_specular;
	DoubleColor attenuation;
	shared_ptr<PDF> pdf_ptr;
};


class Material
{
public:
	Material() {}
	Material(const Vec3& kd, const Vec3& ks, double ns);
	virtual DoubleColor emitted(const Ray& r, const hit_record& rec, double u, double v, const Vec3& p)const
	{
		return DoubleColor(0, 0, 0);
	}
	virtual bool scatter(const Ray& r, const hit_record& rec, scatter_record& srec)const
	{	/*
		double spe = Ks.length(), dif = Kd.length();
		double rate = spe / (spe + dif); default_random_engine e;
		uniform_real_distribution<double> u(0.0, 1.0);

		if(u(e)<rate)
		{
			srec.is_specular = true;
			srec.attenuation = Ks.ScalarProduct(rec.color);
			srec.pdf_ptr = make_shared<cosine_pdf>(rec.normal);
		}else
		{
			srec.is_specular = false;
			srec.attenuation = Kd.ScalarProduct(rec.color);
		}
		*/

		srec.is_specular = false;
		srec.attenuation = Kd.ScalarProduct(rec.color);
		srec.pdf_ptr = make_shared<Cosine_PDF>(rec.normal);
		return true;
	}
	virtual double scattering_pdf(const Ray& r_in, const hit_record& rec, Ray& scatter)const
	{
		auto cosine = dot(rec.normal, unit_vector(scatter.GetDirection()));
		return cosine < 0 ? 0 : cosine / PI;
	}
	~Material()
	{

	}
public:
	//shared_ptr<texture> texture;
	Vec3 Kd, Ks;
	double Ns;
};

class Diffuse_Light :public Material
{
public:
	Diffuse_Light(const Vec3& kd, const Vec3& ks, const Vec3& le, double ns);

	DoubleColor emitted(const Ray& r, const hit_record& rec, double u, double v, const Vec3& p)const override
	{
		return rec.color.ScalarProduct(Le);
	}
	bool scatter(const Ray& r, const hit_record& rec, scatter_record& srec)const override
	{
		return false;
	}
	double scattering_pdf(const Ray& r_in, const hit_record& rec, Ray& scatter)const override
	{
		return 0;
	}
public:
	Vec3 Le;
};
#endif



