#pragma once
#include "gdt/math/AffineSpace.h"
#include "Sphere.h"
// #include "tinyxml2.h"
#include <string>

using namespace gdt;

struct TriangleMesh
{
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    // material data:
    std::string material_name;
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
OptixModel *loadOBJ(const std::string &objFile);

void placeReceiver(Sphere sphere, OptixModel *model, vec3f cameraPosition, float rotation);

void place_receiver_half(HalfSphere side, OptixModel *model, vec3f cameraPosition, bool is_left, float rotation);

void place_receiver_half(HalfSphere side, OptixModel *model, vec3f cameraPosition, bool is_left);

// float get_absorption(int material_id, tinyxml2::XMLDocument &xml_dict);
