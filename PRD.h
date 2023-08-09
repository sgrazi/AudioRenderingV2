#pragma once
#include "optix7.h"

struct __align__(64) PRD
{
    float energy;					    // 4
    float distance;					    // 4
	glm::vec3 curr_position;			// 3 x 4 = 12
	glm::vec3 direction;				// 3 x 4 = 12
	int recursion_depth;				// 4
    glm::vec3 color;				    // 3 x 4 = 12
    // total 48, sobran 16
};
