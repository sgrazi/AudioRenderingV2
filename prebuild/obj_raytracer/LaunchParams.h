#pragma once
#include "optix7.h"
#include <glm/glm.hpp>
#include <string>

struct TriangleMeshSBTData
{
    glm::vec3 pos;
    glm::vec3 *vertex;
    glm::ivec3 *index;
    float mat_absorption;
};

struct Material
{
    std::string name;
    float mat_absorption;
};

struct LaunchParams
{
    const float BASE_POWER = 300.0f;

    int size_x;
    int size_y;
    int size_z;

    glm::vec3 emitter_position;
    glm::vec3 sphere_center;

    OptixTraversableHandle traversable;

    float base_power, energy_thres;
    unsigned int max_bounces;

    int sample_rate;

    bool isMono;
    int ir_length;
    float *ir_left;
    float *ir_right;
};
