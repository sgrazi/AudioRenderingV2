#pragma once
#include "gdt/math/AffineSpace.h"
#include <vector>

using namespace gdt;

struct Material
{
    const char* name;
    float ac_absorption;
};

struct TriangleMesh
{
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    // material data:
    vec3f diffuse;
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

void placeCamera(OptixModel *model, vec3f cameraPosition);