#include "image.h"
#include "shader.h"
#include "raytracer.h"
#include "model.h"
#include <ctime>
#include <SDL2/SDL.h>
#include <thread>

#include "stb_image.h"

#if OBJECT==0
const char* path = "obj/cornellbox/cornellbox.obj";
const char* path_base = "obj/cornellbox";
const char* scene_path = "";
const char* path_result = "result/cornellbox.png";
#elif OBJECT==1
const char* path = "obj/car/car.obj";
const char* path_base = "obj/car";
#if CAMERA==1
const char* scene_path = "obj/car/environment_dusk.hdr";
const char* path_result = "result/car_dusk.png";
#else
const char* scene_path = "obj/car/environment_day.hdr";
const char* path_result = "result/car_day.png";
#endif
#elif OBJECT ==2
const char* path = "./obj/diningroom/diningroom.obj";
const char* path_base = "./obj/diningroom";
const char* scene_path = "obj/diningroom/environment.hdr";
const char* path_result = "result/diningroom.png";
#else

const char* path = "./obj/teapot/teapot.obj";
const char* path_base = "./obj/teapot";
const char* scene_path = "obj/teapot/environment.hdr";
const char* path_result = "result/teapot.png";

#endif

//const char* path0 = "obj/boggie/body.obj";
//const char* path1 = "obj/african_head/african_head.obj";
//const char* path2 = "obj/test2.obj";
//const char* path3 = "obj/test1.obj";
//const char* path4 = "obj/scene2.obj";
//const char* path5 = "obj/dragon.obj";

const unsigned int Width = 1920, Height = 1024;
float zNear = -0.1, zFar = -100.0;
float deltaTime = 0.0f;

const int samples_per_pixel = 2048;
const int bounce_per_pixel = 8;

void Raster();
void RayTrace();

int main(int argc, char** argv) {
#if VERSION == 0
	RayTrace();
#else
	Raster();
#endif
	return 0;
}

#if VERSION == 0
void RayTrace()
{
	Tracer tracer(Width, Height, samples_per_pixel);
	Model modelA(path, path_base);
	if (strcmp(scene_path, "") != 0)
		modelA.set_scene_path(scene_path);
	/*int width, height, depth;
	const auto tex_data = stbi_load(scene_path, &width, &height, &depth, 0);
	vector<stbi_uc> image;
	for (int i = 0; i < width * height * depth;i++)
		image.push_back(tex_data[i]);
	
	
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			tracer.screen.set(i, height-j, image[(i + j * width) * 3], image[(i + j * width) * 3 + 1], image[(i + j * width) * 3 + 2]);
	while (true)
	{
		tracer.show();
	}*/
	
	bool quit = true;
	clock_t start, end;

	std::thread t([&]()
		{
			tracer.tracing(modelA, bounce_per_pixel,path_result);
		});

	while (quit)
	{
		//start = clock();
		tracer.show();
		//end = clock();
		//deltaTime = double(end - start) / CLOCKS_PER_SEC;
		quit = tracer.screen.processEvents(deltaTime);
		//printf("time= %lf s FPS= %lf\n", deltaTime, 1 / deltaTime);
	}
	tracer.stop = true;
	t.join();
}
#else

void Raster()
{
	clock_t start, end;
	Shader shader(Width, Height, 3);
	//const char* path0 = "D:/ZJUfile/G1/CG/CGHW1/CGHW1/Project1/obj/boggie/body.obj";
	//const char* path1 = "D:/ZJUfile/G1/CG/CGHW1/CGHW1/Project1/obj/african_head/african_head.obj";
	//const char* path2 = "D:/ZJUfile/G1/CG/CGHW1/CGHW1/Project1/obj/test2.obj";
	//const char* path3 = "D:/ZJUfile/G1/CG/CGHW1/CGHW1/Project1/obj/test1.obj";
	//const char* path4 = "D:/ZJUfile/G1/CG/CGHW1/CGHW1/Project1/obj/scene2.obj";
	//const char* path5 = "D:/ZJUfile/G1/CG/CGHW1/CGHW1/Project1/obj/dragon.obj";


	//加载模型
	Model modelA(path);
	//modelA.addModel(path4);
	//modelA.addModel(path1);
	//modelA.addModel(path5);

	modelA.SetFace();
	modelA.InitOct(Height, Width);
	modelA.SetABC(zFar, zNear);
	shader.HZB->SetABC(zFar, zNear);
	float deep = -2.0;
	float shape = 1.f;
	bool quit = true;
	bool changed = true;

	mat4 viewportv = viewport(-(int)Width / 2, -(int)Height / 2, Width, Height);
	while (quit)
	{
		start = clock();
		mat4 projection = perspective(radians(shader.screen.camera.Zoom), (float)Width / (float)Height, zNear, zFar);
		mat4 view = shader.screen.camera.GetViewMatrix();
		mat4 model2(1.0f);
		model2 = translate(model2, Vec3(0.0, 0.0, deep));
		model2 = scale(model2, Vec3(1.0, 1.0, 1.0f));
		mat4 model(1.0f);
		model = translate(model, Vec3(2.0, 1.0, -2.0));
		model = scale(model, Vec3(shape, shape, shape));
		modelA.transfer(0, viewportv, projection, view, model2);
		modelA.transfer(1, viewportv, projection, view, model);
		modelA.transfer(2, viewportv, projection, view, model2);
		modelA.transfer(3, viewportv, projection, view, model);
		modelA.Draw(&shader, changed);

		shader.show();
		end = clock();
		deltaTime = double(end - start) / CLOCKS_PER_SEC;
		quit = shader.screen.processEvents(deltaTime, modelA.useTree, shader.useHiZbuffer, changed);
		printf(modelA.useTree ? "OCTree &&" : "not use OCTree &&");
		printf(shader.useHiZbuffer ? "HierachyZbuffer\n" : "Zbuffer\n");
		printf("time= %lf s FPS= %lf\n", deltaTime, 1 / deltaTime);
	}
}

#endif
