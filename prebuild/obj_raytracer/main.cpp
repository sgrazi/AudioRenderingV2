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
#include <ctime>
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
#include "CircularBuffer.h"
#include "Utils.h"

using namespace std;
#define INPUT_SAMPLE_RATE 44100			// default input sample rate
#define INPUT_BUFFER_LENGTH 4096		// default buffer length
std::mutex audio_critical_section;

void full_render(bool isLive, std::mutex *output_buffer_mutex)
{
	AudioRenderer *renderer = Context::get_audio_renderer();
	OptixModel *scene = Context::get_optix_model();
	Sphere sphere = *Context::get_sphere();
	Camera camera = *Context::get_camera();
	gdt::vec3f camera_central_point = gdt::vec3f(camera.Position.x, camera.Position.y, camera.Position.z);

	if (!isLive) {
		AudioFile<float>* audio = Context::get_audio_file();
		size_t len_of_audio = audio->samples[0].size();
		size_t size_of_audio = sizeof(float) * len_of_audio;
		float* outputBuffer_left = Context::get_output_buffer_left();
		float* outputBuffer_right = Context::get_output_buffer_right();
		audio_critical_section.lock();
		renderer->full_render_cycle(output_buffer_mutex, sphere, scene, camera_central_point, camera.globalAngle, audio->samples[0].data(), size_of_audio, outputBuffer_left, outputBuffer_right);
		audio_critical_section.unlock();
	}
	else {
		audio_critical_section.lock();
		placeReceiver(sphere, scene, camera_central_point, camera.globalAngle);
		renderer->setSphereCenterInOptix(glm::vec3(camera_central_point.x, camera_central_point.y, camera_central_point.z));
		renderer->render();
		audio_critical_section.unlock();
	}

	Context::set_is_rendering(false);
};

int audioHandler(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
				double streamTime, RtAudioStreamStatus status, void *userData)
{
	unsigned int i;
	double *buffer = (double *)outputBuffer;
	if (status)
		cout << "Stream underflow detected!" << endl;
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

int audioHandlerWithMic(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
						 double streamTime, RtAudioStreamStatus status, void *data)
{
	if (status)
		cout << "Stream over/underflow detected." << endl;
	Context *context = Context::getInstance();
	audioCallbackData *renderData = (audioCallbackData *)data;
	AudioRenderer *renderer = Context::get_audio_renderer();

	double *buffer = (double *)outputBuffer;
	double *ibuffer = (double *)inputBuffer;

	if (!Context::get_is_rendering()) {
		renderer->convoluteLiveInput(ibuffer, INPUT_BUFFER_LENGTH * sizeof(SAMPLE_TYPE), renderData->samplesRecordBuffer);

		float volume = Context::get_volume();
		std::vector<double> circular_buffer = renderData->samplesRecordBuffer->get_and_reset(nBufferFrames * 2);
		double* host_buffer = circular_buffer.data();
		for (int i = 0; i < nBufferFrames * 2; i++)
		{
			if (host_buffer[i] != host_buffer[i]) {
				*buffer++ = 0;
			}
			else {
				*buffer++ = host_buffer[i] * volume;
			}
		}
		circular_buffer.clear();
	}
	else {
		cout << "Buffer is still being processed" << endl;
		for (int i = 0; i < nBufferFrames * 2; i++)
			*buffer++ = 0;
	}
	
	return 0;
}

int audioPlay(RtAudio *dac)
{
	if (dac->getDeviceCount() < 1)
	{
		cout << "\nNo audio devices found!\n";
		exit(0);
	}
	RtAudio::StreamParameters parameters;
	parameters.deviceId = dac->getDefaultOutputDevice();
	parameters.nChannels = 2;	// TODO: Grab the device output channel count
	parameters.firstChannel = 0; // Default to the first audio output device

	AudioFile<float> *audio = Context::get_audio_file();

	unsigned int sampleRate = audio->getSampleRate() / parameters.nChannels;
	unsigned int bufferFrames = 256; // Default to 256 frame buffer, this is the smallest value properly supported

	AudioInfo *audioInfo = new AudioInfo;
	audioInfo->audio = audio;

	RtAudioErrorType checkError = dac->openStream(&parameters, NULL, RTAUDIO_FLOAT64, sampleRate, &bufferFrames, &audioHandler, (void *)audioInfo);
	checkError = dac->startStream();

	return 0;
}

void audioMicPlay(RtAudio* dac, std::mutex* inputBufferMutex)
{
	// Init audio stream
	dac = new RtAudio();

	if (dac->getDeviceCount() < 1)
	{
		cout << "\nNo audio devices found!\n";
		exit(0);
	}

	unsigned int bufferFrames = INPUT_BUFFER_LENGTH, input_channels = 1, output_channels = 2;
	unsigned int sampleRate = INPUT_SAMPLE_RATE;

	RtAudio::StreamParameters inputParameters;
	inputParameters.deviceId = dac->getDefaultInputDevice();
	inputParameters.nChannels = input_channels;
	inputParameters.firstChannel = 0;

	RtAudio::StreamParameters outputParameters;
	outputParameters.deviceId = dac->getDefaultOutputDevice();
	outputParameters.nChannels = output_channels;
	outputParameters.firstChannel = 0;

	// Calculate buffer size in bytes
	unsigned int bufferBytes = bufferFrames * input_channels * sizeof(SAMPLE_TYPE);

	// Create audioData struct on the heap
	audioCallbackData *audioData = new audioCallbackData;
	audioData->bufferFrames = INPUT_BUFFER_LENGTH;
	audioData->samplesRecordBufferSize = sampleRate * input_channels;
	// circular buffer must be longer than IR
	audioData->samplesRecordBuffer = new CircularBuffer<SAMPLE_TYPE>(INPUT_SAMPLE_RATE * Context::get_ir_length_in_seconds());
	audioData->paths = new audioPaths();
	audioData->paths->ptr = NULL;
	audioData->paths->size = 0;
	audioData->volume = 30.0f;
	audioData->inputBufferMutex = inputBufferMutex;

	RtAudioErrorType checkError = dac->openStream(&outputParameters, &inputParameters, RTAUDIO_FLOAT64, sampleRate, &bufferFrames, &audioHandlerWithMic, (void *)audioData);

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
		cout << "Found an error in audio thread" << endl;
		cout << e.what() << endl;
	}
}

void setSpeakerInScene(glm::vec3 posSpeaker)
{
	std::string speakerPath = "../../assets/models/sphere.obj";
	objl::Loader loader;
	bool load_res = loader.LoadFile(speakerPath);
	std::vector<Mesh> * speaker_vector = Context::get_speaker();

	if (load_res)
	{
		for (int i = 0; i < loader.LoadedMeshes.size(); i++)
		{
			objl::Mesh mesh = loader.LoadedMeshes.at(i);
			std::vector<Vertex> vertices;
			std::vector<unsigned int> indices;
			for (int j = 0; j < mesh.Vertices.size(); j++)
			{
				Vertex vertex;
				vertex.position = glm::vec3(mesh.Vertices.at(j).Position.X + posSpeaker.x, mesh.Vertices.at(j).Position.Y + posSpeaker.y, mesh.Vertices.at(j).Position.Z + posSpeaker.z);
				vertex.normal = glm::vec3(mesh.Vertices.at(j).Normal.X, mesh.Vertices.at(j).Normal.Y, mesh.Vertices.at(j).Normal.Z);
				vertex.color = glm::vec3(mesh.MeshMaterial.Kd.X, mesh.MeshMaterial.Kd.Y, mesh.MeshMaterial.Kd.Z);
				vertices.push_back(vertex);
			}
			for (int j = 0; j < mesh.Indices.size(); j++)
			{
				indices.push_back(mesh.Indices.at(j));
			}
			Mesh speaker(vertices, indices);
			speaker_vector->push_back(speaker);
		}
	}
	else
	{ 
		// error
		throw new std::exception("Error occured while trying load speaker mesh");
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
		cout << "Volume set to " << volume << endl;
	}
	if (key == GLFW_KEY_E)
	{
		Camera *camera = Context::get_camera();
		std::vector<Mesh> * speaker_vector = Context::get_speaker();
		AudioRenderer *renderer = Context::get_audio_renderer();
		speaker_vector->pop_back();
		glm::vec3 cameraPosition = glm::vec3(camera->Position.x, camera->Position.y, camera->Position.z);
		setSpeakerInScene(cameraPosition);
		renderer->setEmitterPosInOptix(cameraPosition);
		cout << "Speaker moved to: " << camera->Position.x << ", " << camera->Position.y << ", " << camera->Position.z << endl;
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
		cout << "Manual render finished" << endl;
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
	std::string scene_file_path = Context::get_scene_file_path();

	GLFWwindow *window = glfwCreateWindow(width, height, "Audiorendering V2", NULL, NULL);
	if (window == NULL)
	{
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		throw new std::exception("Unexpected error while trying to create window");
	}
	glfwMakeContextCurrent(window);

	gladLoadGL();
	glfwSetKeyCallback(window, key_callback);

	glViewport(0, 0, width, height);

	// Load obj && initialize Loader
	objl::Loader loader;
	bool load_res = loader.LoadFile(scene_file_path);
	setSpeakerInScene(Context::get_initial_emitter_pos());
	std::vector<Mesh> lights;
	std::vector<Mesh> objects;
	if (load_res)
	{
		for (int i = 0; i < loader.LoadedMeshes.size(); i++)
		{
			objl::Mesh mesh = loader.LoadedMeshes.at(i);
			std::vector<Vertex> vertices;
			std::vector<unsigned int> indices;
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
		cout << "Found an error in screen thread" << endl;
		throw new std::exception("Error occured while trying load scene mesh");
	}

	Shader shaderProgram("../../assets/shaders/default.vert", "../../assets/shaders/default.frag");

	glm::vec4 lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); // Default light color
	glm::vec3 lightPos = glm::vec3(100, 1000, 300); // Default light position

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
	renderer->setThresholds(Context::get_ray_energy_threshold(), Context::get_ray_max_bounces());
	renderer->set_hrtf_absorption_rate(Context::get_hrtf_absorption_rate());
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

	time_t last_render_time;
	bool timer_set = false;
	time(&last_render_time);
	while (!glfwWindowShouldClose(window))
	{
		glClearColor(0.07f, 0.132f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// change window title
		std::string cameraPosition = "X: " + std::to_string(camera.Position.x) + " Y:" + std::to_string(camera.Position.y) + " Z: " + std::to_string(camera.Position.z);
		std::string newTitle("Audiorendering V2 - " + cameraPosition);
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
		float distanceFromLastRenderPosition = distanceP2P(last_render_position, camera_central_point);
		if (distanceFromLastRenderPosition > 0 && !timer_set) {
			time(&last_render_time);
			timer_set = true;
		}
		bool distanceGreaterThanThreshold = distanceFromLastRenderPosition > re_render_distance_threshold;

		// Trigger re render if angle difference is greater than re_render_angle_threshold
		float angleDiff = abs(last_angle - camera.globalAngle);
		if (angleDiff > 180.0f)
			angleDiff = 360.0f - angleDiff;
		bool angleGreaterThanThreshold = angleDiff > re_render_angle_threshold;

		// Checks if the timer was recently set and the time difference is greatear than a second
		bool last_render_time_trigger = (timer_set && difftime(time(NULL), last_render_time) > 1);

		// Re render condition
		bool reRenderCondition = distanceGreaterThanThreshold || angleGreaterThanThreshold || last_render_time_trigger;

		if (reRenderCondition && !Context::get_is_rendering())
		{
			Context::set_is_rendering(true);
			timer_set = false;
			last_angle = camera.globalAngle;
			last_render_position = camera_central_point;
			std::thread rendering_thread(full_render, Context::get_live_flag(), output_buffer_mutex);
			rendering_thread.detach();
		}

		for (int i = 0; i < objects.size(); i++)
			objects.at(i).Draw(shaderProgram, camera);

		std::vector<Mesh> * speaker_vector = Context::get_speaker();
		speaker_vector->back().Draw(shaderProgram, camera);

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
	std::thread screen1(screen, buffer_mutex);
	std::thread audio1(audio, dac, Context::get_live_flag(), buffer_mutex);

	screen1.join();
	audio1.detach();
	// Stop the stream
	RtAudioErrorType checkError = dac->stopStream();
	if (dac->isStreamOpen())
		dac->closeStream();
	delete dac;
}

// El modo experimentacion permite hacer 100 veces el path tracing y obtener estadisticas de la ejecución
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
	renderer->setThresholds(Context::get_ray_energy_threshold(), Context::get_ray_max_bounces());
	renderer->setEmitterPosInOptix(Context::get_initial_emitter_pos());
	renderer->setSphereCenterInOptix(Context::get_camera()->Position);
	renderer->enable_experimentation();

	// Initialize experimentation
	Experimentation* exp = Experimentation::getInstance();

	Context::set_output_buffer_len(size_of_audio);
	
	gdt::vec3f camera_central_point = gdt::vec3f(camera.Position.x, camera.Position.y, camera.Position.z);

	std::vector<double> render_times;
	std::vector<double> convolute_times;
	std::vector<double> convolute_process_times;

	for (int round_number = 0; round_number < 100; round_number++)
	{
		cout << "Starting round: " << round_number << endl;
		auto start_round_time = std::chrono::high_resolution_clock::now();
		
		double* render_time = new double;
		*render_time = 0;
		renderer->set_write_ir_to_file_flag(false);
		renderer->render(render_time);
		
		auto end_render_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> render_duration = end_render_time - start_round_time;
		cout << "Render time: " << render_duration.count() * 1000 << " ms" << endl;

		auto start_convolute_time = std::chrono::high_resolution_clock::now();
		renderer->set_write_output_to_file_flag(false);

		double* convolute_time = new double;
		double* convolute_process_time = new double;
		*convolute_time = 0;
		*convolute_process_time = 0;
		renderer->convoluteAudioFile(audio->samples[0].data(), size_of_audio, outputBuffer_left, outputBuffer_right, convolute_time, convolute_process_time);
		
		auto end_round_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> convolute_duration = end_round_time - start_convolute_time;
		cout << "Convolute time: " << convolute_duration.count() * 1000 << " ms" << endl;

		std::chrono::duration<double> full_duration = end_round_time - start_round_time;
		cout << "Round " << round_number << ": took " << full_duration.count() * 1000 << " ms" << endl;

		render_times.push_back(*render_time);
		convolute_times.push_back(*convolute_time);
		convolute_process_times.push_back(*convolute_process_time);

		delete render_time;
		delete convolute_time;
		delete convolute_process_time;
	}

	std::string experimentationDirectory = ".\\experimentation";
	std::string prefix = "output_ir_left";
	process_files_with_prefix(experimentationDirectory, prefix);

	Experimentation::results();

	// Return execution times
	cout << "Execution Times:" << endl;

	cout << "\tAverage render time: " << std::accumulate(render_times.begin(), render_times.end(), 0.0) / render_times.size() << " ms" << endl;
	cout << "\tMedian render time: " << median(render_times) << " ms" << endl;

	cout << "\tAverage convolute time: " << std::accumulate(convolute_times.begin(), convolute_times.end(), 0.0) / convolute_times.size() << " ms" << endl;
	cout << "\tMedian convolute time: " << median(convolute_times) << " ms" << endl;

	cout << "\tAverage convolute process time: " << std::accumulate(convolute_process_times.begin(), convolute_process_times.end(), 0.0) / convolute_process_times.size() << " ms" << endl;
	cout << "\tMedian convolute process time: " << median(convolute_process_times) << " ms" << endl;

	render_times.clear();
	convolute_times.clear();
	convolute_process_times.clear();
}

int main(int argc, char **argv)
{

	try
	{
		if (argc < 2)
		{
			cerr << "Insufficient parameters" << endl;
			cerr << "Usage" << argv[0] << " <config_path> [experimental_flag]" << endl;
			return 1;
		}

		std::string config_json_path = argv[1];
		bool main_flag = true;
		if (argc > 2)
			main_flag = std::string(argv[2]) == "true";

		// Read config file
		std::ifstream file(config_json_path);
		if (!file)
			throw std::runtime_error("Error: Unable to open the file: " + config_json_path);

		std::ostringstream ss;
		ss << file.rdbuf(); // reading data
		std::string stringConfig = ss.str();
		if (stringConfig.empty())
			throw std::runtime_error("Error: File is empty or read operation failed");

		cJSON *config = cJSON_Parse(stringConfig.c_str());
		if (!config)
			throw std::runtime_error("Error: JSON parsing failed for the provided string");

		bool is_context_loaded = Context::loadContext(config);
		if (!is_context_loaded)
			throw std::runtime_error("Error: Context failed to load from config file");

		cJSON_Delete(config);

		if (main_flag)
			main_workflow();
		else
			experimentation_mode();
	}
	catch (const std::exception &e)
	{
		cerr << "Exception caught: " << e.what() << endl;
		return 1;
	}

	return 0;
}