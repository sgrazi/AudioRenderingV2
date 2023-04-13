#pragma once

#include <string>
#include <vector>

#include <GL/glew.h>

class Mesh{
private:
	bool textured;
public:
	GLuint verticesID, indicesID, vaoID, shaderID, textureID, normalsID;
	unsigned int vertexCount, indexCount;
public:
	Mesh();
	Mesh(std::vector<float> vertices, std::vector<unsigned int> indices, float * normals);
	void setShader(GLuint shaderID); //No se si poner un puntero al shader que usa
	void draw();
	void addTexture(std::vector<float> textcoords);
	~Mesh();
};

/*
La gracia es que se pueda enchufar un conjunto de vertices y decirle que dibuje la primitiva q se te cante
se podrian calcular las normales a huevo o tambien pasarlas en el constructor
*/