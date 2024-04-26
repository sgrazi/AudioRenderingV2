#pragma once
#include "optix7.h"
#include <glm/glm.hpp>

struct __align__(64) PRD
{
    float remaining_factor;				// 4
    float distance;					    // 4
    glm::vec3 sphere_center;            // 3 x 4 = 12
    glm::vec3 prev_position;			// 3 x 4 = 12
    glm::vec3 direction;				// 3 x 4 = 12
    int recursion_depth;				// 4
    // total 48
};
