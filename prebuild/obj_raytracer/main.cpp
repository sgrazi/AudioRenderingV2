#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <thread>
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>
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
#include "Context.h"
#include "cJSON.h"

using namespace std;

struct AudioInfo
{
	AudioFile<float> *audio;
	float *volumen;
};

int saw(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
				double streamTime, RtAudioStreamStatus status, void *userData)
{
	unsigned int i, j;
	double *buffer = (double *)outputBuffer;
	if (status)
		std::cout << "Stream underflow detected!" << std::endl;
	// Write interleaved audio data.
	AudioInfo *audioInfo = (AudioInfo *)userData;
	float volume = Context::get_volume();
	int nextStream = (int)(streamTime * audioInfo->audio->getSampleRate()) % audioInfo->audio->samples.at(0).size();
	for (i = 0; i < nBufferFrames * 2; i++)
	{
		if (i + nextStream >= audioInfo->audio->samples.at(0).size())
			break;
		*buffer++ = (double)audioInfo->audio->samples.at(0).at(i + nextStream) * volume;
	}
	return 0;
}

int audioPlay(RtAudio *dac, AudioFile<float> *audio)
{
	if (dac->getDeviceCount() < 1)
	{
		std::cout << "\nNo audio devices found!\n";
		exit(0);
	}
	RtAudio::StreamParameters parameters;
	parameters.deviceId = dac->getDefaultOutputDevice();
	parameters.nChannels = 2; // tienq ue machear con los channels del audio
	parameters.firstChannel = 0;

	// send AudioFile info to screen thread??? (audio_length, sample_rate)
	// TODO -> check number of channels.

	unsigned int sampleRate = audio->getSampleRate() / parameters.nChannels;
	unsigned int bufferFrames = 256; // 256 sample frames

	AudioInfo *audioInfo = new AudioInfo;
	audioInfo->audio = audio;

	RtAudioErrorType checkError = dac->openStream(&parameters, NULL, RTAUDIO_FLOAT64, sampleRate, &bufferFrames, &saw, (void *)audioInfo);
	checkError = dac->startStream();

	return 0;
}

void audio(RtAudio *dac, AudioFile<float> *audio)
{
	try
	{
		audioPlay(dac, audio);
	}
	catch (const std::exception &e)
	{
		cout << e.what() << endl;
	}
}

void setTransmitter(glm::vec3 posTransmitter)
{
	std::string transmitterPath = "../../assets/models/sphere.obj";
	objl::Loader loader;
	bool load_res = loader.LoadFile(transmitterPath);
	vector<Mesh> *transmitterVector = Context::get_transmitter();

	if (load_res)
	{
		for (int i = 0; i < loader.LoadedMeshes.size(); i++)
		{
			objl::Mesh mesh = loader.LoadedMeshes.at(i);
			vector<Vertex> vertices;
			vector<unsigned int> indices;
			for (int j = 0; j < mesh.Vertices.size(); j++)
			{
				Vertex vertex;
				vertex.position = glm::vec3(mesh.Vertices.at(j).Position.X + posTransmitter.x, mesh.Vertices.at(j).Position.Y + posTransmitter.y, mesh.Vertices.at(j).Position.Z + posTransmitter.z);
				vertex.normal = glm::vec3(mesh.Vertices.at(j).Normal.X, mesh.Vertices.at(j).Normal.Y, mesh.Vertices.at(j).Normal.Z);
				vertex.color = glm::vec3(mesh.MeshMaterial.Kd.X, mesh.MeshMaterial.Kd.Y, mesh.MeshMaterial.Kd.Z);
				vertices.push_back(vertex);
			}
			for (int j = 0; j < mesh.Indices.size(); j++)
			{
				indices.push_back(mesh.Indices.at(j));
			}
			Mesh transmitter(vertices, indices);
			transmitterVector->push_back(transmitter);
		}
	}
	else
	{ // error
		cout << "Failed to transmitter OBJ" << endl;
		throw new exception("B");
	}
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_RELEASE)
		return; // only handle press events
	if (key == GLFW_KEY_V)
	{
		float volume = Context::get_volume();
		if (volume == 0.0f)
			volume = 1.0f;
		else
			volume = 0.0f;
		Context::set_volume(volume);
		cout << "volumen seteado a " << volume << endl;
	}
	if (key == GLFW_KEY_E)
	{
		Camera *camera = Context::get_camera();
		vector<Mesh> *transmitterVector = Context::get_transmitter();
		AudioRenderer *renderer = Context::get_audio_renderer();
		transmitterVector->pop_back();
		glm::vec3 cameraPosition = glm::vec3(camera->Position.x, camera->Position.y, camera->Position.z);
		setTransmitter(cameraPosition);
		renderer->setEmitterPosInOptix(cameraPosition);
		cout << "Emitter set at: " << camera->Position.x << ", " << camera->Position.y << ", " << camera->Position.z << endl;
	}
	if (key == GLFW_KEY_Q)
	{
		Camera *camera = Context::get_camera();
		Sphere sphere = *Context::get_sphere();
		OptixModel *scene = Context::get_optix_model();
		placeReceiver(sphere, scene, gdt::vec3f(camera->Position.x, camera->Position.y, camera->Position.z));
		cout << "Receiver set at: " << camera->Position.x << ", " << camera->Position.y << ", " << camera->Position.z << endl;
	}
	if (key == GLFW_KEY_R)
	{
		AudioRenderer *renderer = Context::get_audio_renderer();
		renderer->render();
		cout << "Rendered" << endl;
	}
}

void screen(AudioFile<float> *audio)
{
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Get context
	unsigned int width = Context::get_scene_width();
	unsigned int height = Context::get_scene_height();
	string scene_file_path = Context::get_scene_file_path();

	GLFWwindow *window = glfwCreateWindow(width, height, "Audiorendering V2", NULL, NULL);
	if (window == NULL)
	{
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		throw new exception("A");
	}
	glfwMakeContextCurrent(window);

	gladLoadGL();
	glfwSetKeyCallback(window, key_callback);

	glViewport(0, 0, width, height);

	// Load obj && initialize Loader
	objl::Loader loader;
	bool load_res = loader.LoadFile(scene_file_path);
	setTransmitter(Context::get_initial_emitter_pos());
	vector<Mesh> lights;
	vector<Mesh> objects;
	if (load_res)
	{
		for (int i = 0; i < loader.LoadedMeshes.size(); i++)
		{
			objl::Mesh mesh = loader.LoadedMeshes.at(i);
			vector<Vertex> vertices;
			vector<unsigned int> indices;
			for (int j = 0; j < mesh.Vertices.size(); j++)
			{
				Vertex vertex;
				vertex.position = glm::vec3(mesh.Vertices.at(j).Position.X, mesh.Vertices.at(j).Position.Y, mesh.Vertices.at(j).Position.Z);
				vertex.normal = glm::vec3(mesh.Vertices.at(j).Normal.X, mesh.Vertices.at(j).Normal.Y, mesh.Vertices.at(j).Normal.Z);
				vertex.color = glm::vec3(mesh.MeshMaterial.Kd.X, mesh.MeshMaterial.Kd.Y, mesh.MeshMaterial.Kd.Z);
				vertices.push_back(vertex);
			}
			for (int j = 0; j < mesh.Indices.size(); j++)
			{
				indices.push_back(mesh.Indices.at(j));
			}
			Mesh object(vertices, indices);
			objects.push_back(object);
		}
	}
	else
	{ // error
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

	// Camera camera(width, height, glm::vec3(0.0f, 0.0f, 0.0f));

	// load material properties
	// tinyxml2::XMLDocument doc;
	// if (doc.LoadFile("../models/materials.xml") != tinyxml2::XML_SUCCESS)
	// {
	// 	throw std::runtime_error("Failed to load material XML file");
	// }

	// Create Optix mesh of same .obj
	OptixModel *scene = Context::get_optix_model();
	Sphere sphere = *Context::get_sphere();
	Camera camera = *Context::get_camera();
	placeReceiver(sphere, scene, gdt::vec3f(camera.Position.x, camera.Position.y, camera.Position.z));

	// AudioRenderer
	uint32_t sample_rate = Context::get_sample_rate();

	unsigned int ir_length_in_seconds = Context::get_ir_length_in_seconds();
	unsigned int output_channels = Context::get_output_channels();

	AudioRenderer *renderer = Context::get_audio_renderer();
	renderer->setBasePower(Context::get_base_power());
	renderer->setThresholds(Context::get_ray_distance_threshold(), Context::get_ray_energy_threshold(), Context::get_ray_max_bounces());
	renderer->setEmitterPosInOptix(Context::get_initial_emitter_pos());
	renderer->render();

	size_t len_of_audio = audio->samples[0].size();
	size_t size_of_audio = sizeof(float) * len_of_audio;
	float *outputBuffer = (float *)malloc(size_of_audio);
	renderer->convolute(audio->samples[0].data(), size_of_audio, outputBuffer);

	while (!glfwWindowShouldClose(window))
	{
		glClearColor(0.07f, 0.132f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// change window title
		string cameraPosition = "X: " + to_string(camera.Position.x) + " Y:" + to_string(camera.Position.y) + " Z: " + to_string(camera.Position.z);
		string newTitle("Audiorendering V2 - " + cameraPosition);
		glfwSetWindowTitle(window, newTitle.c_str());

		camera.Inputs(window);
		camera.updateMatrix(90.0f, 0.1f, 10000.0f);
		camera.Matrix(shaderProgram, "camMatrix");
		Context::set_camera(&camera);

		for (int i = 0; i < objects.size(); i++)
			objects.at(i).Draw(shaderProgram, camera);

		vector<Mesh> *transmitterVector = Context::get_transmitter();
		transmitterVector->back().Draw(shaderProgram, camera);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	shaderProgram.Delete();
	glfwDestroyWindow(window);
	glfwTerminate();
}

int main(int argc, char **argv)
{
	// Initialize context
	// Not currently being used, TO DO
	string configJsonPath;

	if (argc < 2)
	{
		configJsonPath = "config.json";
	}
	else
	{
		configJsonPath = argv[1];
	}

	// Read config file
	ifstream f(configJsonPath);
	if (!f) {
		throw std::runtime_error("Error: Unable to open the file: " + configJsonPath);
	}
	
	ostringstream ss;
	ss << f.rdbuf(); // reading data
	string stringConfig = ss.str();
	if (stringConfig.empty()) {
		throw std::runtime_error("Error: File is empty or read operation failed");
	}
	cJSON* config = cJSON_Parse(stringConfig.c_str());
	if (!config) {
	    throw std::runtime_error("Error: JSON parsing failed for the provided string");
	}

	// renderer_parameters
	const cJSON* cJSON_renderer_parameters = cJSON_GetObjectItem(config, "renderer_parameters");
	// defaults
	float initial_volume = 1.0f;
	unsigned int output_channels = 2;
	unsigned int ir_length_in_seconds = 2;
	unsigned int width = 1366;
	unsigned int height = 768;
	if (cJSON_IsObject(cJSON_renderer_parameters)) {
		cJSON* cJSON_initial_volume = cJSON_GetObjectItem(cJSON_renderer_parameters, "initial_volume");
		if (cJSON_IsNumber(cJSON_initial_volume))
			initial_volume = cJSON_initial_volume->valuedouble;

		const cJSON* cJSON_output_channels = cJSON_GetObjectItem(cJSON_renderer_parameters, "output_channels");
		if (cJSON_IsNumber(cJSON_output_channels))
			output_channels = round(cJSON_output_channels->valuedouble);

		const cJSON* cJSON_ir_length_in_seconds = cJSON_GetObjectItem(cJSON_renderer_parameters, "ir_length_in_seconds");
		if (cJSON_IsNumber(cJSON_ir_length_in_seconds))
			ir_length_in_seconds = round(cJSON_ir_length_in_seconds->valuedouble);

		const cJSON* cJSON_width = cJSON_GetObjectItem(cJSON_renderer_parameters, "width");
		if (cJSON_IsNumber(cJSON_width))
			width = round(cJSON_width->valuedouble);

		const cJSON* cJSON_height = cJSON_GetObjectItem(cJSON_renderer_parameters, "height");
		if (cJSON_IsNumber(cJSON_height))
			height = round(cJSON_height->valuedouble);
	}
	// scene_parameters
	const cJSON* cJSON_scene_parameters = cJSON_GetObjectItem(config, "renderer_parameters");
	// defaults
	string scene_file_path = "../../assets/models/1D_U.obj";
	string audio_file_path = "../../assets/sound_samples/experimento_entrada_16KHz.wav";
	string materials_file_path = "";
	glm::vec3 initial_receiver_pos(-2.5f, 10.0f, 0.0f);
	glm::vec3 initial_emitter_pos(0, 0, 0);
	if (cJSON_IsObject(cJSON_scene_parameters)) {
		const cJSON* cJSON_scene_file_path = cJSON_GetObjectItem(cJSON_scene_parameters, "scene_file_path");
		if (cJSON_IsString(cJSON_scene_file_path))
			scene_file_path = cJSON_scene_file_path->valuestring;

		const cJSON* cJSON_audio_file_path = cJSON_GetObjectItem(cJSON_scene_parameters, "audio_file_path");
		if (cJSON_IsString(cJSON_audio_file_path))
			audio_file_path = cJSON_audio_file_path->valuestring;

		const cJSON* cJSON_materials_file_path = cJSON_GetObjectItem(cJSON_scene_parameters, "materials_file_path");
		if (cJSON_IsString(cJSON_materials_file_path))
			materials_file_path = cJSON_materials_file_path->valuestring;

		const cJSON* cJSON_initial_receiver_pos = cJSON_GetObjectItem(cJSON_scene_parameters, "initial_receiver_pos");
		if (cJSON_IsObject(cJSON_initial_receiver_pos)) {
			cJSON* x = cJSON_GetObjectItem(cJSON_initial_receiver_pos, "x");
			cJSON* y = cJSON_GetObjectItem(cJSON_initial_receiver_pos, "y");
			cJSON* z = cJSON_GetObjectItem(cJSON_initial_receiver_pos, "z");
			if (cJSON_IsNumber(x) && cJSON_IsNumber(y) && cJSON_IsNumber(z))
				initial_receiver_pos = glm::vec3(x->valuedouble, y->valuedouble, z->valuedouble);
		}

		const cJSON* cJSON_initial_emitter_pos = cJSON_GetObjectItem(cJSON_scene_parameters, "initial_emitter_pos");
		if (cJSON_IsObject(cJSON_initial_emitter_pos)) {
			cJSON* x = cJSON_GetObjectItem(cJSON_initial_emitter_pos, "x");
			cJSON* y = cJSON_GetObjectItem(cJSON_initial_emitter_pos, "y");
			cJSON* z = cJSON_GetObjectItem(cJSON_initial_emitter_pos, "z");
			if (cJSON_IsNumber(x) && cJSON_IsNumber(y) && cJSON_IsNumber(z))
				initial_emitter_pos = glm::vec3(x->valuedouble, y->valuedouble, z->valuedouble);
		}
	}

	// pathtracer_parameters
	const cJSON* cJSON_pathtracer_parameters = cJSON_GetObjectItem(cJSON_pathtracer_parameters, "renderer_parameters");
	// defaults
	float base_power = 100.f; 
	glm::vec3 rays(100, 100, 100);
    float ray_distance_threshold = 100.f;
    float ray_energy_threshold = 0.f;
    unsigned int ray_max_bounces = 10;
	if (cJSON_IsObject(cJSON_pathtracer_parameters)) {
		cJSON* cJSON_base_power = cJSON_GetObjectItem(cJSON_renderer_parameters, "base_power");
		if (cJSON_IsNumber(cJSON_base_power))
			base_power = cJSON_base_power->valuedouble;

		const cJSON* cJSON_rays_size = cJSON_GetObjectItem(cJSON_scene_parameters, "rays");
		if (cJSON_IsObject(cJSON_rays_size)) {
			cJSON* x = cJSON_GetObjectItem(cJSON_rays_size, "x");
			cJSON* y = cJSON_GetObjectItem(cJSON_rays_size, "y");
			cJSON* z = cJSON_GetObjectItem(cJSON_rays_size, "z");
			if (cJSON_IsNumber(x) && cJSON_IsNumber(y) && cJSON_IsNumber(z))
				rays = glm::vec3(x->valuedouble, y->valuedouble, z->valuedouble);
		}
		
		cJSON* cJSON_ray_distance_threshold = cJSON_GetObjectItem(cJSON_renderer_parameters, "ray_distance_threshold");
		if (cJSON_IsNumber(cJSON_ray_distance_threshold))
			ray_distance_threshold = cJSON_ray_distance_threshold->valuedouble;
		
		cJSON* cJSON_ray_energy_threshold = cJSON_GetObjectItem(cJSON_renderer_parameters, "ray_energy_threshold");
		if (cJSON_IsNumber(cJSON_ray_energy_threshold))
			ray_energy_threshold = cJSON_ray_energy_threshold->valuedouble;
		
		const cJSON* cJSON_ray_max_bounces = cJSON_GetObjectItem(cJSON_renderer_parameters, "ray_max_bounces");
		if (cJSON_IsNumber(cJSON_ray_max_bounces))
			ray_max_bounces = round(cJSON_ray_max_bounces->valuedouble);
	}

	cJSON_Delete(config);

	// Setup context

	Context *context = Context::getInstance();
	context->set_volume(initial_volume);
	context->set_ir_length_in_seconds(ir_length_in_seconds);
	context->set_output_channels(output_channels);
	context->set_scene_width(width);
	context->set_scene_height(height);
	context->set_scene_file_path(scene_file_path);
	context->set_audio_file_path(audio_file_path);
	context->set_material_file_path(materials_file_path);
	context->set_ray_distance_threshold(ray_distance_threshold);
	context->set_ray_energy_threshold(ray_energy_threshold);
	context->set_initial_emitter_pos(initial_emitter_pos);
	vector<Mesh> *transmitterVector = new vector<Mesh>;
	context->set_transmitter(transmitterVector);
	Camera *camera = new Camera(width, height, initial_receiver_pos);
	context->set_camera(camera);
	Sphere *sphere = new Sphere();
	context->set_sphere(sphere);
	OptixModel *scene = loadOBJ(scene_file_path);
	context->set_optix_model(scene);
	RtAudio *dac = new RtAudio();
	AudioFile<float> *audio_file = new AudioFile<float>;
	try
	{
		audio_file->load(audio_file_path);
	}
	catch (const std::exception &)
	{
		return 1;
	}

	uint32_t sample_rate = audio_file->getSampleRate();
	context->set_sample_rate(sample_rate);

	placeReceiver(*sphere, scene, gdt::vec3f(camera->Position.x, camera->Position.y, camera->Position.z));

	AudioRenderer *renderer = new AudioRenderer(scene, ir_length_in_seconds, output_channels, sample_rate);
	context->set_audio_renderer(renderer);

	thread screen1(screen, audio_file);
	thread audio1(audio, dac, audio_file);

	screen1.join();
	audio1.detach();
	// Stop the stream
	RtAudioErrorType checkError = dac->stopStream();
	// if (dac.isStreamOpen())
	dac->closeStream();
	delete dac;

	return 0;
}