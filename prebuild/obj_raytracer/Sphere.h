#pragma once
#include "3rdParty/tiny_obj_loader.h"
#include "gdt/math/AffineSpace.h"
#include "HalfSphere.h"

class Sphere
{
public:
    tinyobj::attrib_t original_attributes;
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    Sphere(HalfSphere *left_side, HalfSphere *right_side);
    HalfSphere *left_side;
    HalfSphere *right_side;

    HalfSphere get_left_side();
    HalfSphere get_right_side();
};