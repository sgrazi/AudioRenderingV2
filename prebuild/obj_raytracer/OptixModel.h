#pragma once
#include "gdt/math/AffineSpace.h"
// #include "tinyxml2.h"
#include <vector>

using namespace gdt;

struct TriangleMesh
{
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    // material data:
    vec3f diffuse;
    float material_absorption;
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

// OptixModel *loadOBJ(const std::string &objFile, tinyxml2::XMLDocument &xml_dict);
OptixModel *loadOBJ(const std::string &objFile, int &xml_dict);

void placeCamera(OptixModel *model, vec3f cameraPosition);

// float get_absorption(int material_id, tinyxml2::XMLDocument &xml_dict);