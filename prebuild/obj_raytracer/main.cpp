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
#include <mutex>
#include <stdexcept>
#include <mutex>
#include <functional>
#include <windows.h>
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
#include "Experimentation.h"
#include "cJSON.h"
#include "HalfSphere.h"
#include "CircularBuffer.h"

using namespace std;
#define SAMPLE_TYPE double
#define inputSampleRate 44100 // inventando un sample rate de 44.1khz
#define inputBufferLen 4096		// inventado too

struct audioPaths
{
	void *ptr;
	int size;
};

struct audioCallbackData
{
	int bufferFrames;
	int pos;
	int samplesRecordBufferSize;
	CircularBuffer<SAMPLE_TYPE> *samplesRecordBuffer;
	audioPaths *paths;
	float volume;
	std::mutex* inputBufferMutex;
};

struct AudioInfo
{
	AudioFile<float> *audio;
	float *volumen;
};

float distanceP2P(gdt::vec3f p1, gdt::vec3f p2)
{
	return std::sqrt(std::pow((p2.x - p1.x), 2) + std::pow((p2.y - p1.y), 2) + std::pow((p2.z - p1.z), 2));
}

void full_render(bool testing, std::mutex *output_buffer_mutex)
{
	AudioRenderer *renderer = Context::get_audio_renderer();
	OptixModel *scene = Context::get_optix_model();
	Sphere sphere = *Context::get_sphere();
	Camera camera = *Context::get_camera();
	gdt::vec3f camera_central_point = gdt::vec3f(camera.Position.x, camera.Position.y, camera.Position.z);

	if (!testing) {
		AudioFile<float>* audio = Context::get_audio_file();
		size_t len_of_audio = audio->samples[0].size();
		size_t size_of_audio = sizeof(float) * len_of_audio;
		float* outputBuffer_left = Context::get_output_buffer_left();
		float* outputBuffer_right = Context::get_output_buffer_right();

		renderer->full_render_cycle(output_buffer_mutex, sphere, scene, camera_central_point, camera.globalAngle, audio->samples[0].data(), size_of_audio, outputBuffer_left, outputBuffer_right);
	}
	else {
		output_buffer_mutex->lock();
		placeReceiver(sphere, scene, camera_central_point, camera.globalAngle);
		renderer->setSphereCenterInOptix(glm::vec3(camera_central_point.x, camera_central_point.y, camera_central_point.z));
		renderer->render();
		output_buffer_mutex->unlock();
	}

	Context::set_is_rendering(false);
};

int saw(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
				double streamTime, RtAudioStreamStatus status, void *userData)
{
	unsigned int i;
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
	return 0;
}

// nBufferFrames es inputBufferLen
// len(inputBuffer) es inputBufferLen
// len(outputBuffer) es inputBufferLen * 2
int sawMicro(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
						 double streamTime, RtAudioStreamStatus status, void *data)
{
	if (status)
		std::cout << "Stream over/underflow detected." << std::endl;
	Context *context = Context::getInstance();
	audioCallbackData *renderData = (audioCallbackData *)data;
	AudioRenderer *renderer = Context::get_audio_renderer();

	double *buffer = (double *)outputBuffer;
	double *ibuffer = (double *)inputBuffer;
	std::mutex* inputBufferMutex = renderData->inputBufferMutex;
	if (!Context::get_is_rendering()) {
		renderer->convoluteLiveInput(ibuffer, inputBufferLen * sizeof(SAMPLE_TYPE), renderData->samplesRecordBuffer);

		float volume = Context::get_volume();
		int start = renderData->samplesRecordBuffer->head;
		int length = renderData->samplesRecordBuffer->length;
		for (int i = 0; i < nBufferFrames * 2; i++)
		{
			int index = (start + i) % length;
			*buffer++ = renderData->samplesRecordBuffer->buffer[index] * 100 * volume;
			renderData->samplesRecordBuffer->buffer[index] = 0;
		}
		renderData->samplesRecordBuffer->head = (renderData->samplesRecordBuffer->head + (nBufferFrames * 2)) % length;
	}
	else {
		for (int i = 0; i < nBufferFrames * 2; i++)
			*buffer++ = 0;
	}
	
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
	parameters.nChannels = 2;		 // tiene que matchear con los channels del audio
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

void audioMicPlay(RtAudio* dac, std::mutex* inputBufferMutex)
{
	// Init audio stream
	dac = new RtAudio();

	if (dac->getDeviceCount() < 1)
	{
		std::cout << "\nNo audio devices found!\n";
		exit(0);
	}

	unsigned int bufferFrames = inputBufferLen, input_channels = 1, output_channels = 2;
	unsigned int sampleRate = inputSampleRate;

	RtAudio::StreamParameters inputParameters;
	inputParameters.deviceId = dac->getDefaultInputDevice();
	inputParameters.nChannels = input_channels;
	inputParameters.firstChannel = 0;

	RtAudio::StreamParameters outputParameters;
	outputParameters.deviceId = dac->getDefaultOutputDevice();
	outputParameters.nChannels = output_channels;
	outputParameters.firstChannel = 0;

	RtAudio::StreamOptions options;

	// Calculate buffer size in bytes
	unsigned int bufferBytes = bufferFrames * input_channels * sizeof(SAMPLE_TYPE);

	// Create audioData struct on the heap
	audioCallbackData *audioData = new audioCallbackData;
	audioData->bufferFrames = inputBufferLen;
	audioData->pos = 0;
	audioData->samplesRecordBufferSize = sampleRate * input_channels;
	// circular buffer must be longer than IR
	audioData->samplesRecordBuffer = new CircularBuffer<SAMPLE_TYPE>(inputSampleRate * (Context::get_ir_length_in_seconds() + 2));
	audioData->paths = new audioPaths();
	audioData->paths->ptr = NULL;
	audioData->paths->size = 0;
	audioData->volume = 30.0f;
	audioData->inputBufferMutex = inputBufferMutex;

	RtAudioErrorType checkError = dac->openStream(&outputParameters, &inputParameters, RTAUDIO_FLOAT64, sampleRate, &bufferFrames, &sawMicro, (void *)audioData, &options);

	checkError = dac->startStream();
}

void audio(RtAudio *dac, bool isMic, std::mutex* inputBufferMutex)
{
	try
	{
		if (isMic)
		{
			audioMicPlay(dac, inputBufferMutex);
		}
		else
		{
			audioPlay(dac);
		}
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
		cout << "Volumen seteado a " << volume << endl;
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
		cout << "Emisor colocado en: " << camera->Position.x << ", " << camera->Position.y << ", " << camera->Position.z << endl;
	}
	if (key == GLFW_KEY_R)
	{
		Camera *camera = Context::get_camera();
		Sphere sphere = *Context::get_sphere();
		OptixModel *scene = Context::get_optix_model();
		gdt::vec3f camera_central_position = gdt::vec3f(camera->Position.x, camera->Position.y, camera->Position.z);
		placeReceiver(sphere, scene, camera_central_position, camera->globalAngle);

		AudioRenderer *renderer = Context::get_audio_renderer();
		renderer->render();

		if (!Context::get_live_flag())
		{
			AudioFile<float> *audio = Context::get_audio_file();
			size_t len_of_audio = audio->samples[0].size();
			size_t size_of_audio = sizeof(float) * len_of_audio;
			float *output_buffer_left = Context::get_output_buffer_left();
			float *output_buffer_right = Context::get_output_buffer_right();

			renderer->convoluteAudioFile(audio->samples[0].data(), size_of_audio, output_buffer_left, output_buffer_right);
			Context::set_output_buffer_left(output_buffer_left);
			Context::set_output_buffer_right(output_buffer_right);
			Context::set_output_buffer_len(size_of_audio);
		}

		Context::set_last_render_position(camera_central_position);
		cout << "Rendereado" << endl;
	}
	if (key == GLFW_KEY_P)
	{
		AudioRenderer *renderer = Context::get_audio_renderer();
		renderer->render();
		renderer->set_write_ir_to_file_flag(true);
		renderer->set_write_output_to_file_flag(true);
	}
}

void screen(std::mutex *output_buffer_mutex)
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

	AudioRenderer *renderer = Context::get_audio_renderer();
	renderer->setMonoOutput(Context::get_is_mono());
	renderer->setBasePower(Context::get_base_power());
	renderer->setThresholds(Context::get_ray_distance_threshold(), Context::get_ray_energy_threshold(), Context::get_ray_max_bounces());
	renderer->setEmitterPosInOptix(Context::get_initial_emitter_pos());
	renderer->setSphereCenterInOptix(Context::get_camera()->Position);
	renderer->render();

	size_t size_of_audio;
	float *outputBuffer_left;
	float *outputBuffer_right;
	AudioFile<float> *audio = Context::get_audio_file();
	if (!Context::get_live_flag())
	{
		size_t len_of_audio = audio->samples[0].size();
		size_t size_of_audio = sizeof(float) * len_of_audio;
		float *outputBuffer_left = Context::get_output_buffer_left();
		float *outputBuffer_right = Context::get_output_buffer_right();

		renderer->convoluteAudioFile(audio->samples[0].data(), size_of_audio, outputBuffer_left, outputBuffer_right);

		Context::set_output_buffer_left(outputBuffer_left);
		Context::set_output_buffer_right(outputBuffer_right);
		Context::set_output_buffer_len(size_of_audio);
	}

	gdt::vec3f last_render_position = Context::get_last_render_position();
	float re_render_distance_threshold = Context::get_re_render_distance_threshold();
	float re_render_angle_threshold = Context::get_re_render_angle_threshold();
	glm::vec3 last_orientation = camera.Orientation;
	float last_angle = 0.0f;
	Context::set_is_rendering(false);

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

		if (last_orientation != camera.Orientation)
		{
			camera.calculate_global_angle();
			last_orientation = camera.Orientation;
		}

		// Trigger re render if distance difference is greater than re_render_distance_threshold
		bool distanceGreaterThanThreshold = distanceP2P(last_render_position, camera_central_point) > re_render_distance_threshold;

		// Trigger re render if angle difference is greater than re_render_angle_threshold
		float angleDiff = abs(last_angle - camera.globalAngle);
		if (angleDiff > 180.0f)
			angleDiff = 360.0f - angleDiff;
		bool angleGreaterThanThreshold = angleDiff > re_render_angle_threshold;

		// Re render condition
		bool reRenderCondition = distanceGreaterThanThreshold || angleGreaterThanThreshold;

		if (reRenderCondition && !Context::get_is_rendering())
		{
			Context::set_is_rendering(true);
			last_angle = camera.globalAngle;
			last_render_position = camera_central_point;
			thread rendering_thread(full_render, Context::get_live_flag(), output_buffer_mutex);
			rendering_thread.detach();
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

void main_workflow() {
	std::mutex* buffer_mutex = new std::mutex();
	RtAudio* dac = new RtAudio();
	thread screen1(screen, buffer_mutex);
	thread audio1(audio, dac, Context::get_live_flag(), buffer_mutex);

	screen1.join();
	audio1.detach();
	// Stop the stream
	RtAudioErrorType checkError = dac->stopStream();
	if (dac->isStreamOpen())
		dac->closeStream();
	delete dac;
}

void process_file(const std::string& filePath) {
	// Open the file
	std::ifstream file(filePath);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << filePath << std::endl;
		return;
	}

	string fileName = filePath;
	size_t pos = filePath.find_last_of("/\\");
	if (pos != std::string::npos) {
		fileName = filePath.substr(pos + 1);
	}


	FileData fileData;
	fileData.name = fileName;

	std::string line;
	double max = 0;
	while (std::getline(file, line)) {
		double value = std::atof(line.c_str());
		if (max < value) max = value;
	}

	fileData.maximum_value = max;

	Experimentation::add_file_data(fileData);

	file.close();
}

void process_files_with_prefix(const std::string& directoryPath, const std::string& prefix) {
	std::string searchPath = directoryPath + "\\" + prefix + "*";
	WIN32_FIND_DATA findFileData;
	HANDLE hFind = FindFirstFile(searchPath.c_str(), &findFileData);

	if (hFind == INVALID_HANDLE_VALUE) {
		std::cerr << "Could not open directory: " << directoryPath << std::endl;
		return;
	}

	do {
		std::string fileName = findFileData.cFileName;
		if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			std::string filePath = directoryPath + "\\" + fileName;
			process_file(filePath);
		}
	} while (FindNextFile(hFind, &findFileData) != 0);

	FindClose(hFind);
}

void experimentation_mode() {
	OptixModel* scene = Context::get_optix_model();
	Sphere sphere = *Context::get_sphere();
	Camera camera = *Context::get_camera();
	AudioFile<float>* audio = Context::get_audio_file();
	size_t len_of_audio = audio->samples[0].size();
	size_t size_of_audio = sizeof(float) * len_of_audio;
	float* outputBuffer_left = Context::get_output_buffer_left();
	float* outputBuffer_right = Context::get_output_buffer_right();

	placeReceiver(sphere, scene, gdt::vec3f(camera.Position.x, camera.Position.y, camera.Position.z), camera.globalAngle);

	uint32_t sample_rate = Context::get_sample_rate();

	unsigned int ir_length_in_seconds = Context::get_ir_length_in_seconds();

	AudioRenderer* renderer = Context::get_audio_renderer();
	renderer->setMonoOutput(Context::get_is_mono());
	renderer->setBasePower(Context::get_base_power());
	renderer->setThresholds(Context::get_ray_distance_threshold(), Context::get_ray_energy_threshold(), Context::get_ray_max_bounces());
	renderer->setEmitterPosInOptix(Context::get_initial_emitter_pos());
	renderer->setSphereCenterInOptix(Context::get_camera()->Position);
	renderer->enable_experimentation();

	// Initialize experimentation
	Experimentation* exp = Experimentation::getInstance();

	Context::set_output_buffer_len(size_of_audio);
	
	gdt::vec3f camera_central_point = gdt::vec3f(camera.Position.x, camera.Position.y, camera.Position.z);

	for (int counter = 0; counter < 100; counter++)
	{
		cout << "Starting round: " << counter << endl;
		auto start_round_time = std::chrono::high_resolution_clock::now();
		
		renderer->set_write_ir_to_file_flag(true);
		renderer->render();
		
		auto end_render_time = std::chrono::high_resolution_clock::now();
		chrono::duration<double> render_duration = end_render_time - start_round_time;
		std::cout << "Render time: " << render_duration.count() * 1000 << " ms" << endl;

		/*auto start_convolute_time = std::chrono::high_resolution_clock::now();
		renderer->set_write_output_to_file_flag(true);
		renderer->convoluteAudioFile(audio->samples[0].data(), size_of_audio, outputBuffer_left, outputBuffer_right);*/
		
		auto end_round_time = chrono::high_resolution_clock::now();
		/*chrono::duration<double> convolute_duration = end_round_time - start_convolute_time;
		std::cout << "Convolute time: " << convolute_duration.count() * 1000 << " ms" << endl;*/

		chrono::duration<double> full_duration = end_round_time - start_round_time;
		std::cout << "Round " << counter << ": took " << full_duration.count() * 1000 << " ms" << endl;
	}

	std::string experimentationDirectory = ".\\experimentation";
	std::string prefix = "output_ir_left";  // Change this to your desired prefix

	process_files_with_prefix(experimentationDirectory, prefix);

	Experimentation::results();
}

int main(int argc, char **argv)
{

	try
	{
		// Initialize context
		string configJsonPath;

		if (argc < 2)
			configJsonPath = "../../config.json";
		else
			configJsonPath = argv[1];

		bool main = false;
		if (argc > 3)
			main = argv[2] == "true";

		// Read config file
		ifstream file(configJsonPath);
		if (!file)
			throw std::runtime_error("Error: Unable to open the file: " + configJsonPath);

		ostringstream ss;
		ss << file.rdbuf(); // reading data
		string stringConfig = ss.str();
		if (stringConfig.empty())
			throw std::runtime_error("Error: File is empty or read operation failed");

		cJSON *config = cJSON_Parse(stringConfig.c_str());
		if (!config)
			throw std::runtime_error("Error: JSON parsing failed for the provided string");

		bool is_context_loaded = Context::loadContext(config);
		if (!is_context_loaded)
			throw std::runtime_error("Error: Context failed to load from config file");

		cJSON_Delete(config);

		if (main)
			main_workflow();
		else
			experimentation_mode();
	}
	catch (const exception &e)
	{
		cerr << "Exception caught: " << e.what() << endl;
		return 1;
	}

	return 0;
}