#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "gdt/math/AffineSpace.h"
#include <glm/glm.hpp>
#include <vector>


glm::vec3 gdt2glm(gdt::vec3f vector);

float distanceP2P(gdt::vec3f p1, gdt::vec3f p2);

double median(std::vector<double> values);
#endif
