#include "SceneObject.h"

SceneObject::SceneObject(glm::vec3 pos, float size, OBJProperites props) {
	this->mesh = new Mesh(props.vertices, props.indices, props.normals);
	this->pos = pos;
	this->size = size;
}

void SceneObject::draw() {
	this->mesh->draw();
}

glm::mat4x4 SceneObject::getModelMatrix() {
	return glm::translate(this->pos) * glm::scale(glm::vec3(this->size));
}

SceneObject::~SceneObject(){

}