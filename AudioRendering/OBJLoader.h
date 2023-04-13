#pragma once

#include "tiny_obj_loader.h"

typedef struct OBJProperites {
	std::vector<float> vertices;
	std::vector<unsigned int> indices;
	float * normals;
}OBJProperites;

OBJProperites loadOBJ(std::string file_name);