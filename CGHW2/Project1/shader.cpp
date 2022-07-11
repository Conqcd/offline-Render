#include "shader.h"

Shader::Shader()
{
    width = 800;
    height = 600;
    HZB=new HierarchyZbuffer(width,height);
    useHiZbuffer = false;
}

Shader::Shader(int _width, int _height,int bpp)
{
    width = _width;
    height = _height;
    HZB = new HierarchyZbuffer(width, height);
    screen.init(width, height);
    useHiZbuffer = false;
}



bool Shader::drawTriangle(const Vec3& v1, const Vec3& v2, const Vec3& v3, const AABB3& box, Color color,bool changed)
{

    if (box._min.z > box._max.z || box._max.z >= 1.0 || box._min.z <= -1.0)
        return false;
    if (useHiZbuffer&& !HZB->Culling4(box, true))
    {
		return false;
    }
    int x = box._min.x, y = box._min.y, max_x = box._max.x, max_y = box._max.y;
    float z;
    Vec3 temp1, temp2, bc_screen;
    Vec2 point;
    bool change = false;
    for (; x <= max_x; x++) {
        //#pragma omp parallel for
        for (y = box._min.y; y <= max_y; y++) {
            point.set(x, y);
            barycentric(v1, v2, v3, point, temp1, temp2, bc_screen);
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
            z = bc_screen.x * v1.z + bc_screen.y * v2.z + bc_screen.z * v3.z;
            if (HZB->LessThanZ(z, x, y))
            {
                screen.set(x, y, color.raw[0], color.raw[1], color.raw[2]);
                change = true;
            }
        }
    }
    if (useHiZbuffer&&changed)
		HZB->Update2(box._min.x, box._min.y, box._max.x, box._max.y);
    return change;
}

Shader::~Shader()
{
    delete HZB;
}

void Shader::Switch()
{
    useHiZbuffer = !useHiZbuffer;
}


void Shader::show()
{
    screen.update();
    screen.clear();
}
