#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "Mesh.h"
#include "OBJLoader.h"

class SceneObject {
public:
	Mesh * mesh;
	glm::vec3 pos;
	float size;

public:
	SceneObject(glm::vec3 pos, float size, OBJProperites props);
	void draw();
	glm::mat4x4 getModelMatrix();
	~SceneObject();
};