#include "Sphere.h"

Sphere::Sphere() {
    const std::string objFile = "../../assets/models/sphere2.obj";
    const std::string mtlDir = objFile.substr(0, objFile.rfind('/'));
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK = tinyobj::LoadObj(&attributes,
                                   &shapes,
                                   &materials,
                                   &err,
                                   &err,
                                   objFile.c_str(),
                                   mtlDir.c_str(),
                                   /* triangulate */ true);
    if (!readOK)
    {
        throw std::runtime_error("Could not read sphere OBJ model from " + objFile + " : " + err);
    }

    if (materials.empty())
        throw std::runtime_error("could not parse materials ...");

    this->attributes = attributes;
    this->materials = materials;
    this->original_attributes = attributes;
    this->shapes = shapes;
}