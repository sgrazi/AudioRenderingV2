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

	//Load obj && initialize Loader
	objl::Loader loader;
	bool load_res = loader.LoadFile("sphere.obj");
	GLfloat* vertices;
	GLuint* indices;
	unsigned int vertices_sz;
	unsigned int indices_sz;
	if (load_res)
	{
		vertices_sz = loader.LoadedVertices.size() * 6; // 6 coordinates per vertex
		indices_sz = loader.LoadedIndices.size();
		vertices = new GLfloat[vertices_sz];
		for (int i = 0; i < loader.LoadedVertices.size(); i++) {
			vertices[6 * i] = loader.LoadedVertices.at(i).Position.X;
			vertices[6 * i + 1] = loader.LoadedVertices.at(i).Position.Y;
			vertices[6 * i + 2] = loader.LoadedVertices.at(i).Position.Z;
			vertices[6 * i + 3] = loader.LoadedVertices.at(i).Normal.X;
			vertices[6 * i + 4] = loader.LoadedVertices.at(i).Normal.Y;
			vertices[6 * i + 5] = loader.LoadedVertices.at(i).Normal.Z;
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
	// Vertices coordinates
	//GLfloat vertices[] =
	//{ //     COORDINATES     /        COLORS          /    TexCoord   /        NORMALS       //
	//	-0.5f, 0.0f,  0.5f,     0.0f, -1.0f, 0.0f, // Bottom side
	//	-0.5f, 0.0f, -0.5f,     0.0f, -1.0f, 0.0f, // Bottom side
	//	 0.5f, 0.0f, -0.5f,     0.0f, -1.0f, 0.0f, // Bottom side
	//	 0.5f, 0.0f,  0.5f,     0.0f, -1.0f, 0.0f, // Bottom side

	//	-0.5f, 0.0f,  0.5f,     -0.8f, 0.5f,  0.0f, // Left Side
	//	-0.5f, 0.0f, -0.5f,     -0.8f, 0.5f,  0.0f, // Left Side
	//	 0.0f, 0.8f,  0.0f,     -0.8f, 0.5f,  0.0f, // Left Side

	//	-0.5f, 0.0f, -0.5f,     0.0f, 0.5f, -0.8f, // Non-facing side
	//	 0.5f, 0.0f, -0.5f,     0.0f, 0.5f, -0.8f, // Non-facing side
	//	 0.0f, 0.8f,  0.0f,     0.0f, 0.5f, -0.8f, // Non-facing side

	//	 0.5f, 0.0f, -0.5f,     0.8f, 0.5f,  0.0f, // Right side
	//	 0.5f, 0.0f,  0.5f,     0.8f, 0.5f,  0.0f, // Right side
	//	 0.0f, 0.8f,  0.0f,     0.8f, 0.5f,  0.0f, // Right side

	//	 0.5f, 0.0f,  0.5f,     0.0f, 0.5f,  0.8f, // Facing side
	//	-0.5f, 0.0f,  0.5f,     0.0f, 0.5f,  0.8f, // Facing side
	//	 0.0f, 0.8f,  0.0f,     0.0f, 0.5f,  0.8f  // Facing side
	//};

	//// Indices for vertices order
	//GLuint indices[] =
	//{
	//	0, 1, 2, // Bottom side
	//	0, 2, 3, // Bottom side
	//	4, 6, 5, // Left side
	//	7, 9, 8, // Non-facing side
	//	10, 12, 11, // Right side
	//	13, 15, 14 // Facing side
	//};

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

	GLfloat lightVertices[] =
	{ //     COORDINATES     //
		-0.1f, -0.1f,  0.1f,
		-0.1f, -0.1f, -0.1f,
		 0.1f, -0.1f, -0.1f,
		 0.1f, -0.1f,  0.1f,
		-0.1f,  0.1f,  0.1f,
		-0.1f,  0.1f, -0.1f,
		 0.1f,  0.1f, -0.1f,
		 0.1f,  0.1f,  0.1f
	};

	GLuint lightIndices[] =
	{
		0, 1, 2,
		0, 2, 3,
		0, 4, 7,
		0, 7, 3,
		3, 7, 6,
		3, 6, 2,
		2, 6, 5,
		2, 5, 1,
		1, 5, 4,
		1, 4, 0,
		4, 5, 6,
		4, 6, 7
	};

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

	VAO1.LinkAttrib(VBO1, 0, 3, GL_FLOAT, 6 * sizeof(GLfloat), (void*)0);
	VAO1.LinkAttrib(VBO1, 1, 3, GL_FLOAT, 6 * sizeof(GLfloat), (void*)(3 * sizeof(float)));// definir el stride
	//VAO1.LinkAttrib(VBO1, 1, 3, GL_FLOAT, 6 * sizeof(float), (void*)(3 * sizeof(float)));

	// Good practice to unbind all to prevent accidentally modifying them
	VAO1.Unbind();
	VBO1.Unbind();
	EBO1.Unbind();

	// Shader for light cube
	Shader lightShader("light.vert", "light.frag");
	// Generates Vertex Array Object and binds it
	VAO lightVAO;
	lightVAO.Bind();
	// Generates Vertex Buffer Object and links it to vertices
	VBO lightVBO(lightVertices, sizeof(lightVertices));
	// Generates Element Buffer Object and links it to indices
	EBO lightEBO(lightIndices, sizeof(lightIndices));
	// Links VBO attributes such as coordinates and colors to VAO
	lightVAO.LinkAttrib(lightVBO, 0, 3, GL_FLOAT, 3 * sizeof(float), (void*)0);
	// Unbind all to prevent accidentally modifying them
	lightVAO.Unbind();
	lightVBO.Unbind();
	lightEBO.Unbind();

	glm::vec4 lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	glm::vec3 lightPos = glm::vec3(10.5f, 10.5f, 10.5f);
	glm::mat4 lightModel = glm::mat4(1.0f);
	lightModel = glm::translate(lightModel, lightPos);

	glm::vec3 pyramidPos = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::mat4 pyramidModel = glm::mat4(1.0f);
	pyramidModel = glm::translate(pyramidModel, pyramidPos);

	lightShader.Activate();
	glUniformMatrix4fv(glGetUniformLocation(lightShader.ID, "model"), 1, GL_FALSE, glm::value_ptr(lightModel));
	glUniform4f(glGetUniformLocation(lightShader.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
	shaderProgram.Activate();
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(pyramidModel));
	glUniform4f(glGetUniformLocation(shaderProgram.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
	glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightPos"), lightPos.x, lightPos.y, lightPos.z);

	GLuint timeID = glGetUniformLocation(shaderProgram.ID, "time");

	// Enables the Depth Buffer
	glEnable(GL_DEPTH_TEST);
	
	Camera camera(width, height, glm::vec3(0.0f, 0.0f, 2.0f));

	while (!glfwWindowShouldClose(window)) {
		glClearColor(0.07f, 0.132f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		camera.Inputs(window);
		camera.updateMatrix(45.0f, 0.1f, 100.0f);

		shaderProgram.Activate();
		glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), camera.Position.x, camera.Position.y, camera.Position.z);

		camera.Matrix(shaderProgram, "camMatrix");

		clock_t time = clock();
		glUniform1f(timeID, time);
		VAO1.Bind();
		glDrawElements(GL_TRIANGLES, sizeof(GLuint) * indices_sz /sizeof(int), GL_UNSIGNED_INT, 0);

		lightShader.Activate();
		camera.Matrix(lightShader, "camMatrix");
		lightVAO.Bind();
		glDrawElements(GL_TRIANGLES, sizeof(lightIndices) / sizeof(int), GL_UNSIGNED_INT, 0);


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