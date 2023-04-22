#include "SceneObject.h"

SceneObject::SceneObject(glm::vec3 pos, float size, OBJProperites props) {
	printf("1111111111111111111");
	this->mesh = new Mesh(props.vertices, props.indices, props.normals);
	printf("1111111111111111111");
	this->pos = pos;
	printf("1111111111111111111");
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