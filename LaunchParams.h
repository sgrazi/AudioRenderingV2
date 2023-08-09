#pragma once
#include "optix7.h"
#include <glm/glm.hpp>

struct TriangleMeshSBTData
{
    glm::vec3 color;
    glm::vec3 pos;
    glm::vec3*vertex;
    glm::ivec3*index;
    uint32_t mat;
};

struct LaunchParams
{
    struct
    {
        uint32_t *colorBuffer;
        glm::ivec2 size;
    } frame;

    struct
    {
        glm::vec3 position;
        glm::vec3 direction;
        glm::vec3 horizontal;
        glm::vec3 vertical;
    } camera;
    
    OptixTraversableHandle traversable;
    
    glm::vec3 origin_pos;
    float dist_thres, energy_thres;
    float* other;
};

