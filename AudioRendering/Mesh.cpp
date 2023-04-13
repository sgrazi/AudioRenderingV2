#include "pch.h"
#include "Mesh.h"
#include <exception>
#include <iostream>
#include<fstream>
#include<string>
#include<iterator>

#include <glm/gtc/type_ptr.hpp>

int MESH_WIDTH = 500;
int MESH_LENGTH = 500;

Mesh::Mesh(){

	std::vector<float> positions;

	positions.push_back(-5.0f); positions.push_back(-5.0f); positions.push_back(0.0f);
	positions.push_back(5.0f); positions.push_back(-5.0f); positions.push_back(0.0f);
	positions.push_back(5.0f); positions.push_back(5.0f); positions.push_back(0.0f);
	positions.push_back(5.0f); positions.push_back(5.0f); positions.push_back(0.0f);
	positions.push_back(-5.0f); positions.push_back(5.0f); positions.push_back(0.0f);
	positions.push_back(-5.0f); positions.push_back(-5.0f); positions.push_back(0.0f);

	glGenVertexArrays(1, &vaoID);
	glBindVertexArray(vaoID);


	glGenBuffers(1, &verticesID);
	glBindBuffer(GL_ARRAY_BUFFER, this->verticesID);
	glBufferData(GL_ARRAY_BUFFER, positions.size()*sizeof(float), &positions[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	textured = false;
}

Mesh::Mesh(std::vector<float> vertices, std::vector<unsigned int> indices, float * normals) {
	glGenVertexArrays(1, &vaoID);
	glBindVertexArray(vaoID);

	glGenBuffers(1, &this->verticesID);
	glBindBuffer(GL_ARRAY_BUFFER, this->verticesID);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	this->vertexCount = vertices.size();

	glGenBuffers(1, &this->indicesID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->indicesID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

	this->indexCount = indices.size();

	glGenBuffers(1, &this->normalsID);
	glBindBuffer(GL_ARRAY_BUFFER, this->normalsID);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), normals, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	textured = false;
}

void Mesh::draw() {
	glBindVertexArray(vaoID);
	//El 0 indica que es el primer parametro del "in" del shader, el 1 el siguiente, etc
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	if(textured) glEnableVertexAttribArray(2);
	//glDrawArrays(GL_TRIANGLES,0,6);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->indicesID);
	glDrawElements(
		GL_TRIANGLES,      // mode
		this->indexCount,  // count
		GL_UNSIGNED_INT,   // type
		(void*)0           // element array buffer offset
	);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	if(textured) glDisableVertexAttribArray(2);
}

void Mesh::addTexture(std::vector<float> textcoords) {
	glBindVertexArray(vaoID);
	glGenBuffers(1, &textureID);
	glBindBuffer(GL_ARRAY_BUFFER, this->textureID);
	glBufferData(GL_ARRAY_BUFFER, textcoords.size() * sizeof(float), &textcoords[0], GL_STATIC_DRAW);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

	textured = true;
}

void Mesh::setShader(GLuint shaderID) {
	this->shaderID = shaderID;
}

Mesh::~Mesh() {
	glDeleteBuffers(1, &this->verticesID);
	glDeleteBuffers(1, &this->indicesID);
	glDeleteBuffers(1, &this->normalsID);
	if(textured)glDeleteBuffers(1, &this->textureID);
	glDeleteVertexArrays(1, &vaoID);
}
