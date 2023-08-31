#pragma once
#include "optix7.h"
#include <glm/glm.hpp>
#include <unordered_map>

struct TriangleMeshSBTData
{
    glm::vec3 color;
    glm::vec3 pos;
    glm::vec3* vertex;
    glm::ivec3* index;
    uint32_t mat;
};

struct Material
{
    const char* name;
    float ac_absorption;
};

struct LaunchParams
{
    int size_x;
    int size_y;
    int size_z;

    struct
    {
        glm::vec3 position;
        glm::vec3 direction;
        glm::vec3 horizontal;
        glm::vec3 vertical;
    } camera; //TODO not sure how necessary at this point, we should only care about position

    OptixTraversableHandle traversable;

    std::unordered_map<int, Material> absorption;

    glm::vec3 origin_pos;
    float dist_thres, energy_thres;
    
    int sample_rate;

    int histogram_length;
    float* histogram;
};

