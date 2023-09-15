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
#include "Mesh.h"
#include "AudioFile.h"
#include "RtAudio.h"
#include "OptixModel.h"
#include "AudioRenderer.h"
#include <thread>
#include <filesystem>
#include "Sphere.h"

using namespace std;

const unsigned int width = 1366;
const unsigned int height = 768;
float* volumen = new float(1.0f);
std::string filePath = "../../../assets/models/test.obj";
vector<Mesh> objects;
vector<Mesh> transmitterVector;

Camera camera(width, height, glm::vec3(4.0f, 4.0f, 4.0f));

// Create Optix mesh of same .obj
OptixModel* scene = loadOBJ(filePath);

// AudioRenderer
// TODO modificar cuando se tenga comunicacion entre threads
AudioRenderer* renderer = new AudioRenderer(scene, 256, 256);
glm::ivec2 frameSize(width, height);

struct AudioInfo
{
	AudioFile<float>* audio;
	float* volumen;
};


int saw(void* outputBuffer, void* inputBuffer, unsigned int nBufferFrames,
	double streamTime, RtAudioStreamStatus status, void* userData)
{
	unsigned int i, j;
	double* buffer = (double*)outputBuffer;
	if (status)
		std::cout << "Stream underflow detected!" << std::endl;
	// Write interleaved audio data.
	AudioInfo* audioInfo = (AudioInfo*)userData;

	int nextStream = (int)(streamTime * audioInfo->audio->getSampleRate()) % audioInfo->audio->samples.at(0).size();
	for (i = 0; i < nBufferFrames * 2; i++) {
		if (i + nextStream >= audioInfo->audio->samples.at(0).size()) break;
		*buffer++ = (double)audioInfo->audio->samples.at(0).at(i + nextStream) * (*audioInfo->volumen);
	}
	return 0;
}

int audioPlay(RtAudio* dac)
{
	if (dac->getDeviceCount() < 1) {
		std::cout << "\nNo audio devices found!\n";
		exit(0);
	}
	RtAudio::StreamParameters parameters;
	parameters.deviceId = dac->getDefaultOutputDevice();
	parameters.nChannels = 2; // tienq ue machear con los channels del audio
	parameters.firstChannel = 0;


	AudioFile<float>* audio = new AudioFile<float>;
	const char* file_path = "../../assets/sound_samples/testsound1.wav";
	audio->load(file_path);
	
	// send AudioFile info to screen thread??? (audio_length, sample_rate)

	unsigned int sampleRate = audio->getSampleRate() / audio->getNumChannels();
	unsigned int bufferFrames = 256; // 256 sample frames

	AudioInfo* audioInfo = new AudioInfo;
	audioInfo->audio = audio;
	audioInfo->volumen = volumen;

	RtAudioErrorType checkError = dac->openStream(&parameters, NULL, RTAUDIO_FLOAT64,
		sampleRate, &bufferFrames, &saw, (void*)audioInfo);
	checkError = dac->startStream();

	return 0;
}

void audio(RtAudio* dac) {
	audioPlay(dac);
}

void setTransmitter (glm::vec3 posTransmitter) {
	std::string transmitterPath = "../../../assets/models/sphere.obj";
	objl::Loader loader;
	bool load_res = loader.LoadFile(transmitterPath);

	if (load_res)
	{
		for (int i = 0; i < loader.LoadedMeshes.size(); i++) {
			objl::Mesh mesh = loader.LoadedMeshes.at(i);
			vector<Vertex> vertices;
			vector<unsigned int> indices;
			for (int j = 0; j < mesh.Vertices.size(); j++) {
				Vertex vertex;
				vertex.position = glm::vec3(mesh.Vertices.at(j).Position.X + posTransmitter.x, mesh.Vertices.at(j).Position.Y + posTransmitter.y, mesh.Vertices.at(j).Position.Z + posTransmitter.z);
				vertex.normal = glm::vec3(mesh.Vertices.at(j).Normal.X, mesh.Vertices.at(j).Normal.Y, mesh.Vertices.at(j).Normal.Z);
				vertex.color = glm::vec3(mesh.MeshMaterial.Kd.X, mesh.MeshMaterial.Kd.Y, mesh.MeshMaterial.Kd.Z);
				vertices.push_back(vertex);
			}
			for (int j = 0; j < mesh.Indices.size(); j++) {
				indices.push_back(mesh.Indices.at(j));
			}
			Mesh transmitter(vertices, indices);
			transmitterVector.push_back(transmitter);
		}
	}
	else { // error
		cout << "Failed to transmitter OBJ" << endl;
		throw new exception("B");
	}
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_RELEASE) return; //only handle press events
	if (key == GLFW_KEY_V) {
		renderer->setPos(glm::vec3(10, 10, 10));
		renderer->render();
		renderer->isHit();
		if (*volumen == 0.0f)
			*volumen = 1.0f;
		else
			*volumen = 0.0f;
		cout << "volumen seteado a " << *volumen << endl;
	}
	if (key == GLFW_KEY_E) {
		transmitterVector.pop_back();
		setTransmitter(glm::vec3(camera.Position.x, camera.Position.y, camera.Position.z));
		cout << "transmitter set"  << endl;
	}
}

void screen() {

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(width, height, "Audiorendering V2", NULL, NULL);
	if (window == NULL) {
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		throw new exception("A");
	}
	glfwMakeContextCurrent(window);

	gladLoadGL();
	glfwSetKeyCallback(window, key_callback);

	glViewport(0, 0, width, height);

	//Load obj && initialize Loader
	objl::Loader loader;
	bool load_res = loader.LoadFile(filePath);
	setTransmitter(glm::vec3(0 , 0, 0));
	vector<Mesh> lights;
	if (load_res)
	{
		for (int i = 0; i < loader.LoadedMeshes.size(); i++) {
			objl::Mesh mesh = loader.LoadedMeshes.at(i);
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
	}
	else { // error
		cout << "Failed to load OBJ" << endl;
		throw new exception("B");
	}

	Shader shaderProgram("../../assets/shaders/default.vert", "../../assets/shaders/default.frag");

	glm::vec4 lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	glm::vec3 lightPos = glm::vec3(100, 1000, 300);

	shaderProgram.Activate();
	glUniform4f(glGetUniformLocation(shaderProgram.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
	glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightPos"), lightPos.x, lightPos.y, lightPos.z);

	// Enables the Depth Buffer
	glEnable(GL_DEPTH_TEST);

	Sphere sphere = Sphere();

 //   // Create Optix mesh of same .obj
 //   OptixModel * scene = loadOBJ(filePath);

 //   // AudioRenderer
	//// TODO modificar cuando se tenga comunicacion entre threads
 //   AudioRenderer * renderer = new AudioRenderer(scene, 256, 256);
 //   glm::ivec2 frameSize(width, height);
    renderer->setThresholds(100.0,0.1);
    renderer->setPos(glm::vec3(0.f));
    renderer->setCamera(camera);
    renderer->render();
	renderer->isHit();

	while (!glfwWindowShouldClose(window)) {
		glClearColor(0.07f, 0.132f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//change window title
		string cameraPosition = "X: " + to_string(camera.Position.x) + " Y:" + to_string(camera.Position.y) + " Z: " + to_string(camera.Position.z);
		string newTitle("Audiorendering V2 - " + cameraPosition);
		glfwSetWindowTitle(window, newTitle.c_str());

		camera.Inputs(window);
		camera.updateMatrix(90.0f, 0.1f, 10000.0f);
		camera.Matrix(shaderProgram, "camMatrix");
		placeReceiver(sphere, scene, gdt::vec3f(camera.Position.x, camera.Position.y, camera.Position.z));

		if (*volumen > 0.5) {
			renderer->setCamera(camera);
		}

		for (int i = 0; i < objects.size(); i++)
			objects.at(i).Draw(shaderProgram, camera);

		transmitterVector.back().Draw(shaderProgram, camera);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	shaderProgram.Delete();
	glfwDestroyWindow(window);
	glfwTerminate();
}

int main(int argc, char** argv) {
	// Initialize context
	// Not currently being used, TO DO
	string configJsonPath;

	if (argc < 2) {
		configJsonPath = "config.json";
	}
	else {
		configJsonPath = argv[1];
	}
	RtAudio* dac = new RtAudio();

	thread screen1(screen);
	thread audio1(audio, dac);

	screen1.join();
	audio1.detach();
	// Stop the stream
	RtAudioErrorType checkError = dac->stopStream();
	// if (dac.isStreamOpen()) 
	dac->closeStream();
	delete dac;

	return 0;
}