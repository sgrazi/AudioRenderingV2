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

    OptixTraversableHandle traversable;
    
    vec3f origin_pos;
    float dist_thres, energy_thres;
    float* other;
};

