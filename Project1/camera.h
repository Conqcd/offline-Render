#pragma once

#include"vec.h"
#include"Matrix.h"
#include "ray.h"
#include"common.h"
#include <vector>
#include<cmath>

/*
 *	此处参考LearnOpenGL的相机做法
 *	但是此处View矩阵从左手变成右手系
 */

#if CAMERA == 0

#define FOV 60
//(0,0,2.5) (0,0,0)
#define POSITION Vec3(0.0,0,2.5)
#define LOOKAT Vec3(0,0,0)
#define UP Vec3(0,1,0)

#elif  CAMERA == 1

#define FOV 45
#define POSITION Vec3(5.72,0.12,9.55)
#define LOOKAT Vec3( 5.085,   -0.131,    8.819)
#define UP Vec3( -0.165,    0.968,   -0.189)

#elif  CAMERA == 2

#define FOV 45
#define POSITION Vec3(8.22, -0.61, -9.80)
#define LOOKAT Vec3( 7.514,   -0.702,   -9.097)
#define UP Vec3(-0.065,    0.996,    0.065)

#elif CAMERA == 3

#define FOV 45
#define POSITION Vec3( 0.000,   12.720,   31.850) 
#define LOOKAT Vec3( 0.000,   12.546,   30.865)
#define UP Vec3(0.000,    0.985,   -0.174)

#else

#define FOV 20

#define POSITION Vec3(30.0,12.0,0)
#define LOOKAT Vec3(0,2.5,0)
#define UP Vec3(0,1,0)

#endif


enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

#if VERSION ==0


const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.1f;


class Camera
{
public:

	Vec3 Position;
	Vec3 Front;
	Vec3 Up;
	Vec3 Right;

	Vec3 horizontal;
	Vec3 vertical;
	Vec3 lower_left;

	float Yaw;
	float Pitch;

	float MovementSpeed;
	float MouseSensitivity;
	float Fov;
	double ASPECTION_RATIO;

	GPULINE Camera(double aspect_ratio = 1.0, Vec3 position = POSITION, Vec3 up = UP, float yaw = YAW, float pitch = PITCH) :Front(LOOKAT - POSITION), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Fov(FOV), ASPECTION_RATIO(aspect_ratio)
	{
		Position = position;
		Up = up;
		Yaw = yaw;
		Pitch = pitch;

		updateCameraVectors();

	}
	
	GPULINE void ProcessKeyboard(Camera_Movement direction, float deltaTime)
	{
		float cameraSpeed = 2.5f * deltaTime;
		if (direction == FORWARD)
			Position += cameraSpeed * Front;
		if (direction == BACKWARD)
			Position -= cameraSpeed * Front;
		if (direction == LEFT)
			Position -= Right * cameraSpeed;
		if (direction == RIGHT)
			Position += Right * cameraSpeed;
	}

	GPULINE void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true)
	{
		/*
		xoffset *= MouseSensitivity;
		yoffset *= MouseSensitivity;

		Yaw += xoffset;
		Pitch += yoffset;

		if (constrainPitch)
		{
			if (Pitch >= 89.9f)
				Pitch = 89.9f;
			if (Pitch <= -89.9f)
				Pitch = -89.9f;
		}
		updateCameraVectors();
		*/
	}

	GPULINE void ProcessMouseScroll(float yoffset)
	{
		Fov -= yoffset;
		if (Fov < 1.0f)
			Fov = 1.0f;
		if (Fov > 45.0f)
			Fov = 45.0f;
	}
	
	GPULINE Ray Get_Ray(double u, double v)
	{
		return Ray(Position, lower_left + u * vertical + v * horizontal);
	}
private:
	GPULINE void updateCameraVectors()
	{
		Front = unit_vector(Front);
		Up = unit_vector(Up);
		Right = unit_vector(cross(Front, Up));

		double theta = radians(Fov);
		auto h = tan(theta / 2);
		auto ViewHeight = 2.0 * h;
		auto ViewWidth = ViewHeight * ASPECTION_RATIO;

		horizontal = ViewWidth * Right;
		vertical = ViewHeight * Up;
		lower_left = Front - horizontal / 2 - vertical / 2;
	}
};

#else

const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;


class Camera
{
public:

	Vec3 Position;
	Vec3 Front;
	Vec3 Up;
	Vec3 Right;
	Vec3 WorldUp;

	float Yaw;
	float Pitch;

	float MovementSpeed;
	float MouseSensitivity;
	float Zoom;
	double ASPECTION_RATIO;

	Camera(double aspect_ratio = 1.0, Vec3 position = POSITION, Vec3 up = UP, float yaw = YAW, float pitch = PITCH) :Front(LOOKAT - POSITION), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(FOV), ASPECTION_RATIO(aspect_ratio)
	{
		Position = position;
		WorldUp = up;
		Yaw = yaw;
		Pitch = pitch;

		updateCameraVectors();

	}
	Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch, double aspect_ratio = 1.0) :Front(Vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(FOV), ASPECTION_RATIO(aspect_ratio)
	{
		Position = Vec3(posX, posY, posZ);
		WorldUp = Vec3(upX, upY, upZ);
		Yaw = yaw;
		Pitch = pitch;
		updateCameraVectors();
	}

	mat4 GetViewMatrix()
	{
		mat4 Result(1.0f);

		Result[0][0] = Right.getx();
		Result[0][1] = Right.gety();
		Result[0][2] = Right.getz();
		Result[1][0] = Up.getx();
		Result[1][1] = Up.gety();
		Result[1][2] = Up.getz();
		Result[2][0] = -Front.getx();
		Result[2][1] = -Front.gety();
		Result[2][2] = -Front.getz();
		Result[0][3] = -dot(Right, Position);
		Result[1][3] = -dot(Up, Position);
		Result[2][3] = -dot(-Front, Position);

		return Result;
	}

	void ProcessKeyboard(Camera_Movement direction, float deltaTime)
	{
		float cameraSpeed = 2.5f * deltaTime;
		if (direction == FORWARD)
			Position += cameraSpeed * Front;
		if (direction == BACKWARD)
			Position -= cameraSpeed * Front;
		if (direction == LEFT)
			Position -= Right * cameraSpeed;
		if (direction == RIGHT)
			Position += Right * cameraSpeed;
	}

	void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true)
	{

		xoffset *= MouseSensitivity;
		yoffset *= MouseSensitivity;

		Yaw += xoffset;
		Pitch += yoffset;

		if (constrainPitch)
		{
			if (Pitch >= 89.9f)
				Pitch = 89.9f;
			if (Pitch <= -89.9f)
				Pitch = -89.9f;
		}
		updateCameraVectors();
	}

	void ProcessMouseScroll(float yoffset)
	{
		Zoom -= yoffset;
		if (Zoom < 1.0f)
			Zoom = 1.0f;
		if (Zoom > 45.0f)
			Zoom = 45.0f;
	}

private:
	void updateCameraVectors()
	{
		Vec3 front;
		front.set(cos(radians(Yaw)) * cos(radians(Pitch)), sin(radians(Pitch)), sin(radians(Yaw)) * cos(radians(Pitch)));
		Front = unit_vector(front);
		Right = unit_vector(cross(Front, WorldUp));
		Up = unit_vector(cross(Right, Front));

	}

	Vec3 getGroundDirection()
	{
		return unit_vector(Vec3(Front.getx(), 0.0f, Front.getz()));
	}
};
#endif
