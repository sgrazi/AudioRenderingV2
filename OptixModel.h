#pragma once
#include <glm/glm.hpp>
#include "gdt/math/box.h"
#include <vector>

using namespace gdt;

struct Material
{
    const uint32_t id;
    const char* name;
    float ac_absorption;
};

struct TriangleMesh
{
    std::vector<glm::vec3> vertex;
    std::vector<glm::vec3> normal;
    std::vector<glm::vec2> texcoord;
    std::vector<glm::ivec3> index;

    // material data:
    glm::vec3 diffuse;
    uint32_t materialID;
};

struct OptixModel
{
    ~OptixModel()
    {
        for (auto mesh : meshes)
            delete mesh;
    }

    std::vector<TriangleMesh *> meshes;
    //! bounding box of all vertices in the model
    box3f bounds;
};

OptixModel *loadOBJ(const std::string &objFile);

void placeReceiverInScene(OptixModel *model, glm::vec3 targetPosition);