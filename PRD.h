#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

struct __align__(64) PRD
{
    float energy;					    // 4
    float distance;					    // 4
	vec3f position;				        // 3 x 4 = 12
	vec3f direction;				    // 3 x 4 = 12
	int recursion_depth;				// 4
    vec3f color;				        // 3 x 4 = 12
    // total 48, sobran 16
};
