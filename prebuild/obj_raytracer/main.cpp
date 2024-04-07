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
#include <mutex>
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
#include "HalfSphere.h"

using namespace std;
std::mutex _mutex;

struct AudioInfo
{
	AudioFile<float> *audio;
	float *volumen;
};

float distanceP2P(gdt::vec3f p1, gdt::vec3f p2) {
	return std::sqrt(std::pow((p2.x - p1.x), 2) + std::pow((p2.y - p1.y), 2) + std::pow((p2.z - p1.z), 2));
}

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
	Context *context = Context::getInstance();
	float *outputBufferConvolute_left = context->get_output_buffer_left();
	float *outputBufferConvolute_right = context->get_output_buffer_right();
	_mutex.lock();
	for (i = 0; i < nBufferFrames * 2; i++)
	{
		if (i + nextStream >= context->get_output_buffer_len())
			break;
		// *buffer++ = (double)audioInfo->audio->samples.at(0).at(i + nextStream) * volume;
		if (i % 2 == 0)
		{
			*buffer++ = outputBufferConvolute_left[i + nextStream] * 100 * volume;
		}
		else
		{
			*buffer++ = outputBufferConvolute_right[i + nextStream] * 100 * volume;
		}
	}
	_mutex.unlock();
	return 0;
}

int audioPlay(RtAudio *dac)
{
	if (dac->getDeviceCount() < 1)
	{
		std::cout << "\nNo audio devices found!\n";
		exit(0);
	}
	RtAudio::StreamParameters parameters;
	parameters.deviceId = dac->getDefaultOutputDevice();
	parameters.nChannels = 2;	 // tiene que matchear con los channels del audio
	parameters.firstChannel = 0; // Default audio output

	// TODO -> check number of channels.

	AudioFile<float> *audio = Context::get_audio_file();

	unsigned int sampleRate = audio->getSampleRate() / parameters.nChannels;
	unsigned int bufferFrames = 256; // 256 sample frames

	AudioInfo *audioInfo = new AudioInfo;
	audioInfo->audio = audio;

	RtAudioErrorType checkError = dac->openStream(&parameters, NULL, RTAUDIO_FLOAT64, sampleRate, &bufferFrames, &saw, (void *)audioInfo);
	checkError = dac->startStream();

	return 0;
}

void audio(RtAudio *dac)
{
	try
	{
		audioPlay(dac);
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
	if (key == GLFW_KEY_R)
	{
		_mutex.lock();
		Camera* camera = Context::get_camera();
		Sphere sphere = *Context::get_sphere();
		OptixModel* scene = Context::get_optix_model();
		gdt::vec3f camera_central_position = gdt::vec3f(camera->Position.x, camera->Position.y, camera->Position.z);
		placeReceiver(sphere, scene, camera_central_position, camera->globalAngle);

		AudioRenderer *renderer = Context::get_audio_renderer();
		renderer->render();

		AudioFile<float> *audio = Context::get_audio_file();
		size_t len_of_audio = audio->samples[0].size();
		size_t size_of_audio = sizeof(float) * len_of_audio;
		float *output_buffer_left = Context::get_output_buffer_left();
		float *output_buffer_right = Context::get_output_buffer_right();

		renderer->convolute(audio->samples[0].data(), size_of_audio, output_buffer_left, output_buffer_right, Context::get_output_channels());
		Context::set_output_buffer_left(output_buffer_left);
		Context::set_output_buffer_right(output_buffer_right);
		Context::set_output_buffer_len(size_of_audio);
		Context::set_last_render_position(camera_central_position);
		_mutex.unlock();
		cout << "Rendered" << endl;
	}
	if (key == GLFW_KEY_P)
	{
		AudioRenderer *renderer = Context::get_audio_renderer();
		renderer->render();
		renderer->set_write_ir_to_file_flag(true);
		renderer->set_write_output_to_file_flag(true);
	}
}

void screen()
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

	// Create Optix mesh of same .obj
	OptixModel *scene = Context::get_optix_model();
	Sphere sphere = *Context::get_sphere();
	Camera camera = *Context::get_camera();
	placeReceiver(sphere, scene, gdt::vec3f(camera.Position.x, camera.Position.y, camera.Position.z), camera.globalAngle);

	// AudioRenderer
	uint32_t sample_rate = Context::get_sample_rate();

	unsigned int ir_length_in_seconds = Context::get_ir_length_in_seconds();
	unsigned int output_channels = Context::get_output_channels();

	AudioRenderer *renderer = Context::get_audio_renderer();
	renderer->setBasePower(Context::get_base_power());
	renderer->setThresholds(Context::get_ray_distance_threshold(), Context::get_ray_energy_threshold(), Context::get_ray_max_bounces());
	renderer->setEmitterPosInOptix(Context::get_initial_emitter_pos());
	renderer->render();

	AudioFile<float> *audio = Context::get_audio_file();
	size_t len_of_audio = audio->samples[0].size();
	size_t size_of_audio = sizeof(float) * len_of_audio;
	float *outputBuffer_left = Context::get_output_buffer_left();
	float *outputBuffer_right = Context::get_output_buffer_right();

	renderer->convolute(audio->samples[0].data(), size_of_audio, outputBuffer_left, outputBuffer_right, output_channels);

	Context::set_output_buffer_left(outputBuffer_left);
	Context::set_output_buffer_right(outputBuffer_right);
	Context::set_output_buffer_len(size_of_audio);

	gdt::vec3f last_render_position = Context::get_last_render_position();
	float re_render_distance_threshold = Context::get_re_render_distance_threshold();
	float re_render_angle_threshold = Context::get_re_render_angle_threshold();
	glm::vec3 last_orientation = camera.Orientation;
	float last_angle = 0.0f;

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
		gdt::vec3f camera_central_point = gdt::vec3f(camera.Position.x, camera.Position.y, camera.Position.z);

		if (last_orientation != camera.Orientation) {
			camera.calculate_global_angle();
			last_orientation = camera.Orientation;
		}

		// Trigger re render if distance difference is greater than re_render_distance_threshold
		bool distanceGreaterThanThreshold = distanceP2P(last_render_position, camera_central_point) > re_render_distance_threshold;

		// Trigger re render if angle difference is greater than re_render_angle_threshold
		float angleDiff = abs(last_angle - camera.globalAngle);
		if (angleDiff > 180.0f) angleDiff = 360.0f - angleDiff;
		bool angleGreaterThanThreshold = angleDiff > re_render_angle_threshold;

		// Re render condition
		bool reRenderCondition = distanceGreaterThanThreshold || angleGreaterThanThreshold;

		if (reRenderCondition) {
			last_angle = camera.globalAngle;
			last_render_position = camera_central_point;
			_mutex.lock();
			placeReceiver(sphere, scene, camera_central_point, camera.globalAngle);
			renderer->render();
			renderer->convolute(audio->samples[0].data(), size_of_audio, outputBuffer_left, outputBuffer_right, output_channels);
			_mutex.unlock();
		}

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

	try
	{
		// Initialize context
		string configJsonPath;

		if (argc < 2) configJsonPath = "../../config.json";
		else configJsonPath = argv[1];

		// Read config file
		ifstream file(configJsonPath);
		if (!file) throw std::runtime_error("Error: Unable to open the file: " + configJsonPath);

		ostringstream ss;
		ss << file.rdbuf(); // reading data
		string stringConfig = ss.str();
		if (stringConfig.empty()) throw std::runtime_error("Error: File is empty or read operation failed");

		cJSON *config = cJSON_Parse(stringConfig.c_str());
		if (!config) throw std::runtime_error("Error: JSON parsing failed for the provided string");
		
		bool is_context_loaded = Context::loadContext(config);
		if (!is_context_loaded) throw std::runtime_error("Error: Context failed to load from config file");

		cJSON_Delete(config);

		RtAudio* dac = new RtAudio();
		thread screen1(screen);
		thread audio1(audio, dac);

		screen1.join();
		audio1.detach();
		// Stop the stream
		RtAudioErrorType checkError = dac->stopStream();
		if (dac->isStreamOpen()) dac->closeStream();
		delete dac;
	}
	catch (const exception &e)
	{
		cerr << "Exception caught: " << e.what() << endl;
		return 1;
	}

	return 0;
}