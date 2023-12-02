#pragma once
#include "optix7.h"
#include <glm/glm.hpp>

struct TriangleMeshSBTData
{
    glm::vec3 color;
    glm::vec3 pos;
    glm::vec3 *vertex;
    glm::ivec3 *index;
    float mat_absorption;
};

struct Material
{
    int id;
    const char *name;
    float ac_absorption;
};

struct LaunchParams
{
    int size_x;
    int size_y;
    int size_z;

    glm::vec3 emitter_position;

    OptixTraversableHandle traversable;

    float base_power, dist_thres, energy_thres;
    unsigned int max_bounces;

    int sample_rate;

    int ir_length;
    float *ir;
};
