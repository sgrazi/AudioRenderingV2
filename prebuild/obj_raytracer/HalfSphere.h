#pragma once
#include "3rdParty/tiny_obj_loader.h"
#include "gdt/math/AffineSpace.h"
#include <string>

class HalfSphere
{
public:
    tinyobj::attrib_t original_attributes;
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    HalfSphere(std::string path);
};