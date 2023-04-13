#include "Source.h"
#include <GL/glew.h>

#include "OBJLoader.h"

Source::Source(glm::vec3 position, float radius, std::string file_name) {
	this->pos = position;
	this->sphere_radius = radius;
	OBJProperites props = loadOBJ(file_name);
	this->mesh = new Mesh(props.vertices, props.indices, props.normals);
}

glm::mat4x4 Source::getModelMatrix() {
	return glm::translate(this->pos) * glm::scale(glm::vec3(this->sphere_radius));
}

void Source::draw() {
	this->mesh->draw();
}

Source::~Source() {

}