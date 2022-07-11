#pragma once

#include <SDL2/SDL.h>

#include "camera.h"

struct Color {
	union {
		struct {
			unsigned char b, g, r;
		};
		unsigned char raw[3];
	};

	Color(unsigned char R, unsigned char G, unsigned char B) : b(B), g(G), r(R) {
	}

};

using DoubleColor = Vec3;


class SDLImage
{
protected:
	int width, height;
	SDL_Window* window;
	SDL_Texture* texture;
	SDL_Renderer* renderer;
	Uint32* buffer;
	float last_x, last_y;
public:
	Camera camera;
	SDLImage();
	SDLImage(int w, int h);
	//SDLImage(const SDLImage& img);
	~SDLImage();
	SDLImage& operator =(const SDLImage& img);
	int get_width()
	{
		return width;
	}
	int get_height()
	{
		return height;
	}
	bool init(int w,int h);
	void clear();
	bool processEvents(float deltaTime, bool& set1, bool& set2, bool& changed);
	bool processEvents(float deltaTime);
	void update();
	bool set(int x, int y, Uint8 red, Uint8 green, Uint8 blue);
	bool set(int x, int y, double red, double green, double blue);
	bool set(int x, int y, float red, float green, float blue, int samples_per_pixel);
	void write_picture(char const* filename);
};
