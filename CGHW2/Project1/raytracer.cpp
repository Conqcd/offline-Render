#include "raytracer.h"

#if VERSION==0

Tracer::Tracer()
{
    width = 800;
    height = 600;
}

Tracer::Tracer(int _width, int _height, int spp)
{
    width = _width;
    height = _height;
    samples_per_pixel = spp;
    screen.init(width, height);
}

void Tracer::tracing(const Model& model,int depth,const char* path_result)
{
#if USEGPU==true
    GPU_DataSet gmodel=CreateGPUModel(model,height,width);
	while(!stop)
	{
        RayTracing(gmodel,depth,screen,samples_per_pixel, background);
        screen.write_picture(path_result);
	}
    gmodel.deleteMySelf();
#else
#pragma omp parallel for
    for (int i = height - 1; i >= 0; i--)
    {
        if (stop)
        {
            continue;
        }
        for (int j = 0; j < width; j++)
        {
            if (stop) continue;
            DoubleColor pixel_color(0, 0, 0);
            for (int k = 0; k < samples_per_pixel; k++)
            {
                if (stop) continue;
                double u = (double(i) + random_double()) / (height - 1);
                double v = (double(j) + random_double()) / (width - 1);
                Ray	r = screen.camera.Get_Ray(u, v);
                pixel_color += ray_color(r, background, model, depth);
            }
            screen.set(j, i, pixel_color.getx() / samples_per_pixel, pixel_color.gety() / samples_per_pixel, pixel_color.getz() / samples_per_pixel);
        }
        //printf("%d\n", i);
    }
#endif
}

#if USEGPU==false
DoubleColor Tracer::ray_color(const Ray& r, const DoubleColor& background, const Model& model, int depth)
{
	
    hit_record record;
    if (depth <= 0)	return DoubleColor(0.0, 0.0, 0.0);
    if (!model.hit(0.001, infinity, r, record))
        return background;
    Ray scattered;
    double pdf_value;
    scatter_record srecord;
    DoubleColor emitted = record.mat_ptr->emitted(r, record, record.u, record.v, record.point);

    if (!record.mat_ptr->scatter(r, record, srecord))
        return emitted;

    if (srecord.is_specular)
    {
        return srecord.attenuation.ScalarProduct(ray_color(srecord.specular_ray, background, model, depth - 1));
    }
    scattered = Ray(record.point, srecord.pdf_ptr->generate(), r.GetTime());
    pdf_value = srecord.pdf_ptr->value(scattered.GetDirection());
	return emitted + srecord.attenuation.ScalarProduct(ray_color(scattered, background, model, depth - 1))
        * record.mat_ptr->scattering_pdf(r, record, scattered) / pdf_value;
    /*

    auto light_ptr = (make_shared<hittable_pdf>(lights, record.point));
    mixture_pdf p(light_ptr, srecord.pdf_ptr);

    scattered = Ray(record.point, p.generate(), r.get_Time());
    pdf_value = p.value(scattered.get_Direction());

    return emitted + srecord.attenuation * ray_color(scattered, background, world, depth - 1, lights)
        * record.mat_ptr->scattering_pdf(r, record, scattered) / pdf_value;
	*/
}
#endif

void Tracer::show()
{
    screen.update();
    //screen.clear();
}

#endif