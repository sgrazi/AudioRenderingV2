#include "Sphere.h"

Sphere::Sphere(HalfSphere *left_side, HalfSphere *right_side) {

    this->left_side = left_side;
    this->right_side = right_side;
}

HalfSphere Sphere::get_left_side() 
{
    return *this->left_side;
}

HalfSphere Sphere::get_right_side() 
{
    return *this->right_side;
}