#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include "OBJ_Loader.h"
#include "VAO.h"
#include "VBO.h"
#include "EBO.h"
#include "shaderClass.h"
#include "Camera.h"
using namespace std;

const unsigned int width = 800;
const unsigned int height = 800;

int main() {
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Load obj && initialize Loader
	objl::Loader loader;
	bool load_res = loader.LoadFile("sphere.obj");
	GLfloat* vertices;
	GLuint* indices;
	unsigned int vertices_sz;
	unsigned int indices_sz;
	if (load_res)
	{
		vertices_sz = loader.LoadedVertices.size() * 3; // 3 coordinates per vertex
		indices_sz = loader.LoadedIndices.size();
		vertices = new GLfloat[vertices_sz];
		for (int i = 0; i < loader.LoadedVertices.size(); i++) {
			vertices[3 * i] = loader.LoadedVertices.at(i).Position.X;
			vertices[3 * i + 1] = loader.LoadedVertices.at(i).Position.Y;
			vertices[3 * i + 2] = loader.LoadedVertices.at(i).Position.Z;
			//cout << vertices[3 * i] << " " << vertices[3 * i + 1] << " " << vertices[3 * i + 2] << endl;
		}

		indices = new GLuint[indices_sz];
		for (int i = 0; i < indices_sz; i++) {
			indices[i] = loader.LoadedIndices.at(i);
			//cout << indices[i] << endl;
		}
	} else { // error
		cout << "Failed to load OBJ" << endl;
		return -1;
	}

	//GLfloat vertices[] =
	//{ // COORDINATES           // COLORS
	//	-0.5f, 0.0f,  0.5f,     0.83f, 0.70f, 0.44f,
	//	-0.5f, 0.0f, -0.5f,     0.83f, 0.70f, 0.44f,
	//	 0.5f, 0.0f, -0.5f,     0.83f, 0.70f, 0.44f,
	//	 0.5f, 0.0f,  0.5f,     0.83f, 0.70f, 0.44f,
	//	 0.0f, 0.8f,  0.0f,     0.92f, 0.86f, 0.76f
	//};

	//// Indices for vertices order
	//GLuint indices[] =
	//{
	//	0, 1, 2,
	//	0, 2, 3,
	//	0, 1, 4,
	//	1, 2, 4,
	//	2, 3, 4,
	//	3, 0, 4
	//};

	

	GLFWwindow* window = glfwCreateWindow(width, height, "Raytracer pero del bueno", NULL, NULL);
	if (window == NULL) {
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	gladLoadGL();

	glViewport(0, 0, width, height);

	Shader shaderProgram("default.vert", "default.frag");
	
	VAO VAO1;
	VAO1.Bind();

	VBO VBO1(vertices, sizeof(GLfloat) * vertices_sz);
	EBO EBO1(indices, sizeof(GLuint) * indices_sz);

	VAO1.LinkAttrib(VBO1, 0, 3, GL_FLOAT, 3 * sizeof(GLfloat), (void*)0);
	//VAO1.LinkAttrib(VBO1, 1, 3, GL_FLOAT, 6 * sizeof(float), (void*)(3 * sizeof(float)));

	// Good practice to unbind all to prevent accidentally modifying them
	VAO1.Unbind();
	VBO1.Unbind();
	EBO1.Unbind();

	GLuint timeID = glGetUniformLocation(shaderProgram.ID, "time");

	// Enables the Depth Buffer
	glEnable(GL_DEPTH_TEST);
	
	Camera camera(width, height, glm::vec3(0.0f, 0.0f, 2.0f));

	while (!glfwWindowShouldClose(window)) {
		glClearColor(0.07f, 0.132f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		shaderProgram.Activate();

		camera.Inputs(window);
		camera.Matrix(45.0f, 0.1f, 100.0f, shaderProgram, "camMatrix");

		clock_t time = clock();
		glUniform1f(timeID, time);
		VAO1.Bind();
		glDrawElements(GL_TRIANGLES, sizeof(GLuint) * indices_sz /sizeof(int), GL_UNSIGNED_INT, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	VAO1.Delete();
	VBO1.Delete();
	EBO1.Delete();
	shaderProgram.Delete();
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}