#pragma once
#include "gdt/math/vec.h"
#include "optix7.h"

using namespace gdt;

struct TriangleMeshSBTData
{
    vec3f color;
    vec3f pos;
    vec3f *vertex;
    vec3i *index;
    uint32_t mat;
};

struct LaunchParams
{
    struct
    {
        uint32_t *colorBuffer;
        vec2i size;
    } frame;

    struct
    {
        vec3f position;
        vec3f direction;
        vec3f horizontal;
        vec3f vertical;
    } camera;

    OptixTraversableHandle traversable;
    
    vec3f pos;
    float dist_thres, energy_thres;
};

