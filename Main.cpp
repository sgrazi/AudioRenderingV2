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
#include"Mesh.h"
using namespace std;

const unsigned int width = 800;
const unsigned int height = 800;

//// Vertices coordinates
//Vertex vertices[] =
//{ //               COORDINATES           /            COLORS          /           NORMALS         /       TEXTURE COORDINATES    //
//	Vertex{glm::vec3(-1.0f, 0.0f,  1.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f)},
//	Vertex{glm::vec3(-1.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f)},
//	Vertex{glm::vec3(1.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f)},
//	Vertex{glm::vec3(1.0f, 0.0f,  1.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f)}
//};
//
//// Indices for vertices order
//GLuint indices[] =
//{
//	0, 1, 2,
//	0, 2, 3
//};

int main() {
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(width, height, "Raytracer pero del bueno", NULL, NULL);
	if (window == NULL) {
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	gladLoadGL();

	glViewport(0, 0, width, height);

	//Load obj && initialize Loader
	objl::Loader loader;
	bool load_res = loader.LoadFile("test.obj");
	
	//if (load_res)
	//{
	//	vertices_sz = loader.LoadedVertices.size() * 6; // 6 coordinates per vertex
	//	indices_sz = loader.LoadedIndices.size();
	//	vertices = new GLfloat[vertices_sz];
	//	for (int i = 0; i < loader.LoadedVertices.size(); i++) {
	//		vertices[6 * i] = loader.LoadedVertices.at(i).Position.X;
	//		vertices[6 * i + 1] = loader.LoadedVertices.at(i).Position.Y;
	//		vertices[6 * i + 2] = loader.LoadedVertices.at(i).Position.Z;
	//		vertices[6 * i + 3] = loader.LoadedVertices.at(i).Normal.X;
	//		vertices[6 * i + 4] = loader.LoadedVertices.at(i).Normal.Y;
	//		vertices[6 * i + 5] = loader.LoadedVertices.at(i).Normal.Z;
	//		//cout << vertices[3 * i] << " " << vertices[3 * i + 1] << " " << vertices[3 * i + 2] << endl;
	//	}

	//	indices = new GLuint[indices_sz];
	//	for (int i = 0; i < indices_sz; i++) {
	//		indices[i] = loader.LoadedIndices.at(i);
	//		//cout << indices[i] << endl;
	//	}
	//} else { // error
	//	cout << "Failed to load OBJ" << endl;
	//	return -1;
	//}

	vector<Mesh> objects;
	vector<Mesh> lights;
	//std::vector <Vertex> verts(vertices, vertices + sizeof(vertices) / sizeof(Vertex));
	//std::vector <GLuint> ind(indices, indices + sizeof(indices) / sizeof(GLuint));
	//	// Create floor mesh
	//Mesh object(verts, ind);
	//objects.push_back(object);
	if (load_res)
	{
		for (int i = 0; i < loader.LoadedMeshes.size(); i++) {
			objl::Mesh mesh = loader.LoadedMeshes.at(i);
			if (mesh.MeshMaterial.name != "Luz") {
				vector<Vertex> vertices;
				vector<unsigned int> indices;
				for (int j = 0; j < mesh.Vertices.size(); j++) {
					Vertex vertex;
					vertex.position = glm::vec3(mesh.Vertices.at(j).Position.X, mesh.Vertices.at(j).Position.Y, mesh.Vertices.at(j).Position.Z);
					vertex.normal = glm::vec3(mesh.Vertices.at(j).Normal.X, mesh.Vertices.at(j).Normal.Y, mesh.Vertices.at(j).Normal.Z);
					vertex.color = glm::vec3(mesh.MeshMaterial.Kd.X, mesh.MeshMaterial.Kd.Y, mesh.MeshMaterial.Kd.Z);
					vertices.push_back(vertex);
				}
				for (int j = 0; j < mesh.Indices.size(); j++) {
					indices.push_back(mesh.Indices.at(j));
				}
				Mesh object(vertices, indices);
				objects.push_back(object);
			}
			else {
				vector<Vertex> vertices;
				vector<unsigned int> indices;
				for (int j = 0; j < mesh.Vertices.size(); j++) {
					Vertex vertex;
					vertex.position = glm::vec3(mesh.Vertices.at(j).Position.X, mesh.Vertices.at(j).Position.Y, mesh.Vertices.at(j).Position.Z);
					vertex.normal = glm::vec3(mesh.Vertices.at(j).Normal.X, mesh.Vertices.at(j).Normal.Y, mesh.Vertices.at(j).Normal.Z);
					vertex.color = glm::vec3(mesh.MeshMaterial.Kd.X, mesh.MeshMaterial.Kd.Y, mesh.MeshMaterial.Kd.Z);
					vertices.push_back(vertex);
				}
				for (int j = 0; j < mesh.Indices.size(); j++) {
					indices.push_back(mesh.Indices.at(j));
				}
				Mesh object(vertices, indices);
				lights.push_back(object);
			}
			
		}	
	}
	else { // error
		cout << "Failed to load OBJ" << endl;
		return -1;
	}

	for (int j = 0; j < loader.LoadedMaterials.size(); j++) {
		cout << loader.LoadedMaterials.at(j).name << endl;
	}

	Shader shaderProgram("default.vert", "default.frag");
	Shader lightShader("light.vert", "light.frag");

	glm::vec4 lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	glm::vec3 lightPos = glm::vec3(	);
	glm::mat4 lightModel = glm::mat4(1.0f);


	glm::vec3 objectPos = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::mat4 objectModel = glm::mat4(1.0f);
	objectModel = glm::translate(objectModel, objectPos);

	lightShader.Activate();
	glUniformMatrix4fv(glGetUniformLocation(lightShader.ID, "model"), 1, GL_FALSE, glm::value_ptr(lightModel));
	glUniform4f(glGetUniformLocation(lightShader.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
	shaderProgram.Activate();
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(objectModel));
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

		for (int i = 0; i < objects.size(); i++)
			objects.at(i).Draw(shaderProgram, camera);

		for (int i = 0; i < lights.size(); i++)
			lights.at(i).Draw(lightShader, camera);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	shaderProgram.Delete();
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}