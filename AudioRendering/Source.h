#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "Mesh.h"

class Source {
public:
	glm::vec3 pos;
	float sphere_radius;
	Mesh * mesh;

public:
	Source() {};
	Source(glm::vec3 position, float radius, std::string file_name);
	glm::mat4x4 getModelMatrix();
	void draw();
	~Source();
};