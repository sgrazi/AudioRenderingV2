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

using namespace std;
#define SAMPLE_TYPE double
#define inputSampleRate 44100 //inventando un sample rate de 44.1khz
#define inputBufferLen 4096 //inventado too
std::mutex outputBufferMutex;
std::mutex inputBufferMutex;

struct audioPaths {
    void* ptr;
    int size;
};

struct audioCallbackData {
    int bufferFrames;
    int pos;
    int samplesRecordBufferSize;
    CircularBuffer<SAMPLE_TYPE>* samplesRecordBuffer;
    audioPaths* paths;
    float volume;
};


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
	if (status) std::cout << "Stream over/underflow detected." << std::endl;
	Context *context = Context::getInstance();
	audioCallbackData *renderData = (audioCallbackData *)data;
	AudioRenderer *renderer = Context::get_audio_renderer();

	double *buffer = (double*)outputBuffer;
	double *ibuffer = (double*)inputBuffer;
	inputBufferMutex.lock();
	renderer->convoluteLiveInput(ibuffer, inputBufferLen * sizeof(SAMPLE_TYPE), nBufferFrames * 2 * sizeof(double), renderData->samplesRecordBuffer);
	inputBufferMutex.unlock();
	
	float volume = Context::get_volume();
	int start = renderData->samplesRecordBuffer->head;
	int length = renderData->samplesRecordBuffer->length;
	for (int i = 0; i < nBufferFrames * 2 ; i++) {
		int index = (start + i) % length;
		*buffer++ = renderData->samplesRecordBuffer->buffer[index] * 50 * volume;
		renderData->samplesRecordBuffer->buffer[index] = 0;
	}
	renderData->samplesRecordBuffer->head = (renderData->samplesRecordBuffer->head + (nBufferFrames * 2)) % length;
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
		std::cout << "\nSAMPLE RATE\n";
		std::cout << sampleRate << '\n';
	unsigned int bufferFrames = 256; // 256 sample frames

	AudioInfo *audioInfo = new AudioInfo;
	audioInfo->audio = audio;

	RtAudioErrorType checkError = dac->openStream(&parameters, NULL, RTAUDIO_FLOAT64, sampleRate, &bufferFrames, &saw, (void *)audioInfo);
	checkError = dac->startStream();

	return 0;
}

void audioMicPlay(RtAudio *dac) {
    // Init audio stream
    dac = new RtAudio();

    if (dac->getDeviceCount() < 1) {
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
    audioCallbackData* audioData = new audioCallbackData;
    audioData->bufferFrames = inputBufferLen;
    audioData->pos = 0;
    audioData->samplesRecordBufferSize = sampleRate * input_channels;
    audioData->samplesRecordBuffer = new CircularBuffer<SAMPLE_TYPE>(inputSampleRate * 4);
    audioData->paths = new audioPaths();
    audioData->paths->ptr = NULL;
    audioData->paths->size = 0;
    audioData->volume = 30.0f;

    std::cout << "\nLLAMO OPEN STREAM\n";
    RtAudioErrorType checkError = dac->openStream(&outputParameters, &inputParameters, RTAUDIO_FLOAT64, sampleRate, &bufferFrames, &sawMicro, (void *)audioData, &options);

    std::cout << "\nLLAMO OPEN START STREAM\n";
    checkError = dac->startStream();
}

void audio(RtAudio *dac, bool isMic)
{
	try
	{
		if (isMic) 
		{
			audioMicPlay(dac);
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
		Camera* camera = Context::get_camera();
		Sphere sphere = *Context::get_sphere();
		OptixModel* scene = Context::get_optix_model();
		gdt::vec3f camera_central_position = gdt::vec3f(camera->Position.x, camera->Position.y, camera->Position.z);
		placeReceiver(sphere, scene, camera_central_position);

		AudioRenderer *renderer = Context::get_audio_renderer();
		renderer->render();

		if (!Context::get_live_flag()) {
			AudioFile<float> *audio = Context::get_audio_file();
			size_t len_of_audio = audio->samples[0].size();
			size_t size_of_audio = sizeof(float) * len_of_audio;
			float *output_buffer_left = Context::get_output_buffer_left();
			float *output_buffer_right = Context::get_output_buffer_right();

			renderer->convoluteAudioFile(audio->samples[0].data(), size_of_audio, output_buffer_left, output_buffer_right, Context::get_output_channels());
			Context::set_output_buffer_left(output_buffer_left);
			Context::set_output_buffer_right(output_buffer_right);
			Context::set_output_buffer_len(size_of_audio);
		}

		Context::set_last_render_position(camera_central_position);
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

	size_t size_of_audio;
	float* outputBuffer_left;
	float* outputBuffer_right;
	AudioFile<float>* audio = Context::get_audio_file();
	if (!Context::get_live_flag()){
		size_t len_of_audio = audio->samples[0].size();
		size_t size_of_audio = sizeof(float) * len_of_audio;
		float *outputBuffer_left = Context::get_output_buffer_left();
		float *outputBuffer_right = Context::get_output_buffer_right();

		renderer->convoluteAudioFile(audio->samples[0].data(), size_of_audio, outputBuffer_left, outputBuffer_right, output_channels);

		Context::set_output_buffer_left(outputBuffer_left);
		Context::set_output_buffer_right(outputBuffer_right);
		Context::set_output_buffer_len(size_of_audio);
	}
	
	gdt::vec3f last_render_position = Context::get_last_render_position();
	float re_render_distance_threshold = Context::get_re_render_distance_threshold();

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
		if (distanceP2P(last_render_position, camera_central_point) > re_render_distance_threshold) {
			placeReceiver(sphere, scene, camera_central_point);
			renderer->render();
			if (!Context::get_live_flag()){
				renderer->convoluteAudioFile(audio->samples[0].data(), size_of_audio, outputBuffer_left, outputBuffer_right, output_channels);
			}
			last_render_position = camera_central_point;
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

		if (argc < 2)
		{
			configJsonPath = "../../config.json";
		}
		else
		{
			configJsonPath = argv[1];
		}

		// Read config file
		ifstream f(configJsonPath);
		if (!f)
		{
			throw std::runtime_error("Error: Unable to open the file: " + configJsonPath);
		}

		ostringstream ss;
		ss << f.rdbuf(); // reading data
		string stringConfig = ss.str();
		if (stringConfig.empty())
		{
			throw std::runtime_error("Error: File is empty or read operation failed");
		}
		cJSON *config = cJSON_Parse(stringConfig.c_str());
		if (!config)
		{
			throw std::runtime_error("Error: JSON parsing failed for the provided string");
		}

		// renderer_parameters
		const cJSON *cJSON_renderer_parameters = cJSON_GetObjectItem(config, "renderer_parameters");
		// defaults
		float initial_volume = 0.5f;
		unsigned int output_channels = 2;
		unsigned int ir_length_in_seconds = 2;
		unsigned int width = 1366;
		unsigned int height = 768;
		bool write_ir_to_file_on_render = false;
		bool write_output_to_file_on_render = false;
		float re_render_distance_threshold = 3;
		if (cJSON_IsObject(cJSON_renderer_parameters))
		{
			cJSON *cJSON_initial_volume = cJSON_GetObjectItem(cJSON_renderer_parameters, "initial_volume");
			if (cJSON_IsNumber(cJSON_initial_volume))
				initial_volume = cJSON_initial_volume->valuedouble;

			const cJSON *cJSON_output_channels = cJSON_GetObjectItem(cJSON_renderer_parameters, "output_channels");
			if (cJSON_IsNumber(cJSON_output_channels))
				output_channels = round(cJSON_output_channels->valuedouble);

			const cJSON *cJSON_ir_length_in_seconds = cJSON_GetObjectItem(cJSON_renderer_parameters, "ir_length_in_seconds");
			if (cJSON_IsNumber(cJSON_ir_length_in_seconds))
				ir_length_in_seconds = round(cJSON_ir_length_in_seconds->valuedouble);

			const cJSON *cJSON_width = cJSON_GetObjectItem(cJSON_renderer_parameters, "width");
			if (cJSON_IsNumber(cJSON_width))
				width = round(cJSON_width->valuedouble);

			const cJSON *cJSON_height = cJSON_GetObjectItem(cJSON_renderer_parameters, "height");
			if (cJSON_IsNumber(cJSON_height))
				height = round(cJSON_height->valuedouble);

			const cJSON *cJSON_write_ir_to_file_on_render = cJSON_GetObjectItem(cJSON_renderer_parameters, "write_first_ir_to_file");
			if (cJSON_IsBool(cJSON_write_ir_to_file_on_render))
				write_ir_to_file_on_render = cJSON_IsTrue(cJSON_write_ir_to_file_on_render);

			const cJSON *cJSON_write_output_to_file_on_render = cJSON_GetObjectItem(cJSON_renderer_parameters, "write_first_output_to_file");
			if (cJSON_IsBool(cJSON_write_output_to_file_on_render))
				write_output_to_file_on_render = cJSON_IsTrue(cJSON_write_output_to_file_on_render);
			
			const cJSON *cJSON_re_render_distance_threshold = cJSON_GetObjectItem(cJSON_renderer_parameters, "re_render_distance_threshold");
			if (cJSON_IsBool(cJSON_re_render_distance_threshold))
				re_render_distance_threshold = cJSON_IsTrue(cJSON_re_render_distance_threshold);
				
		}
		// scene_parameters
		const cJSON *cJSON_scene_parameters = cJSON_GetObjectItem(config, "scene_parameters");
		// defaults
		string scene_file_path = "../../assets/models/1D_U.obj";
		string audio_file_path;
		string materials_file_path = "";
		glm::vec3 initial_receiver_pos(-2.5f, 10.0f, 0.0f);
		glm::vec3 initial_emitter_pos(0, 0, 0);
		if (cJSON_IsObject(cJSON_scene_parameters))
		{
			const cJSON *cJSON_scene_file_path = cJSON_GetObjectItem(cJSON_scene_parameters, "scene_file_path");
			if (cJSON_IsString(cJSON_scene_file_path))
				scene_file_path = cJSON_scene_file_path->valuestring;

			const cJSON *cJSON_audio_file_path = cJSON_GetObjectItem(cJSON_scene_parameters, "audio_file_path");
			if (cJSON_IsString(cJSON_audio_file_path))
				audio_file_path = cJSON_audio_file_path->valuestring;

			const cJSON *cJSON_materials_file_path = cJSON_GetObjectItem(cJSON_scene_parameters, "materials_file_path");
			if (cJSON_IsString(cJSON_materials_file_path))
				materials_file_path = cJSON_materials_file_path->valuestring;

			const cJSON *cJSON_initial_receiver_pos = cJSON_GetObjectItem(cJSON_scene_parameters, "initial_receiver_pos");
			if (cJSON_IsObject(cJSON_initial_receiver_pos))
			{
				cJSON *x = cJSON_GetObjectItem(cJSON_initial_receiver_pos, "x");
				cJSON *y = cJSON_GetObjectItem(cJSON_initial_receiver_pos, "y");
				cJSON *z = cJSON_GetObjectItem(cJSON_initial_receiver_pos, "z");
				if (cJSON_IsNumber(x) && cJSON_IsNumber(y) && cJSON_IsNumber(z))
					initial_receiver_pos = glm::vec3(x->valuedouble, y->valuedouble, z->valuedouble);
			}

			const cJSON *cJSON_initial_emitter_pos = cJSON_GetObjectItem(cJSON_scene_parameters, "initial_emitter_pos");
			if (cJSON_IsObject(cJSON_initial_emitter_pos))
			{
				cJSON *x = cJSON_GetObjectItem(cJSON_initial_emitter_pos, "x");
				cJSON *y = cJSON_GetObjectItem(cJSON_initial_emitter_pos, "y");
				cJSON *z = cJSON_GetObjectItem(cJSON_initial_emitter_pos, "z");
				if (cJSON_IsNumber(x) && cJSON_IsNumber(y) && cJSON_IsNumber(z))
					initial_emitter_pos = glm::vec3(x->valuedouble, y->valuedouble, z->valuedouble);
			}
		}

		// pathtracer_parameters
		const cJSON *cJSON_pathtracer_parameters = cJSON_GetObjectItem(config, "pathtracer_parameters");
		// defaults
		float base_power = 100.f;
		glm::vec3 rays(100, 100, 100);
		float ray_distance_threshold = 100.f;
		float ray_energy_threshold = 0.f;
		unsigned int ray_max_bounces = 10;
		vector<Material> materials;
		if (cJSON_IsObject(cJSON_pathtracer_parameters))
		{
			cJSON *cJSON_base_power = cJSON_GetObjectItem(cJSON_renderer_parameters, "base_power");
			if (cJSON_IsNumber(cJSON_base_power))
				base_power = cJSON_base_power->valuedouble;

			const cJSON *cJSON_rays_size = cJSON_GetObjectItem(cJSON_scene_parameters, "rays");
			if (cJSON_IsObject(cJSON_rays_size))
			{
				cJSON *x = cJSON_GetObjectItem(cJSON_rays_size, "x");
				cJSON *y = cJSON_GetObjectItem(cJSON_rays_size, "y");
				cJSON *z = cJSON_GetObjectItem(cJSON_rays_size, "z");
				if (cJSON_IsNumber(x) && cJSON_IsNumber(y) && cJSON_IsNumber(z))
					rays = glm::vec3(x->valuedouble, y->valuedouble, z->valuedouble);
			}

			cJSON *cJSON_ray_distance_threshold = cJSON_GetObjectItem(cJSON_renderer_parameters, "ray_distance_threshold");
			if (cJSON_IsNumber(cJSON_ray_distance_threshold))
				ray_distance_threshold = cJSON_ray_distance_threshold->valuedouble;

			cJSON *cJSON_ray_energy_threshold = cJSON_GetObjectItem(cJSON_renderer_parameters, "ray_energy_threshold");
			if (cJSON_IsNumber(cJSON_ray_energy_threshold))
				ray_energy_threshold = cJSON_ray_energy_threshold->valuedouble;

			const cJSON *cJSON_ray_max_bounces = cJSON_GetObjectItem(cJSON_renderer_parameters, "ray_max_bounces");
			if (cJSON_IsNumber(cJSON_ray_max_bounces))
				ray_max_bounces = round(cJSON_ray_max_bounces->valuedouble);

			const cJSON *cJSON_materials = cJSON_GetObjectItem(cJSON_pathtracer_parameters, "materials");
			const cJSON *cJSON_material = NULL;
			if (cJSON_IsArray(cJSON_materials))
			{
				cJSON_ArrayForEach(cJSON_material, cJSON_materials)
				{
					Material material = Material();
					cJSON *name = cJSON_GetObjectItem(cJSON_material, "name");
					cJSON *mat_absorption = cJSON_GetObjectItem(cJSON_material, "mat_absorption");
					if (cJSON_IsString(name) && cJSON_IsNumber(mat_absorption))
					{
						material.name = name->valuestring;
						material.mat_absorption = mat_absorption->valuedouble;
						// Add material to material list;
						materials.push_back(material);
					}
				}
			}
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
		context->set_material_file_path(materials_file_path);
		context->set_ray_distance_threshold(ray_distance_threshold);
		context->set_ray_energy_threshold(ray_energy_threshold);
		context->set_ray_max_bounces(ray_max_bounces);
		context->set_base_power(base_power);
		// Pos para escuchar por el izquierdo context->set_initial_emitter_pos(glm::vec3(-2.5, 10, -10));
		context->set_initial_emitter_pos(initial_emitter_pos);
		vector<Mesh> *transmitterVector = new vector<Mesh>;
		context->set_transmitter(transmitterVector);
		Camera *camera = new Camera(width, height, initial_receiver_pos);
		context->set_camera(camera);

		gdt::vec3f last_render_position = gdt::vec3f(camera->Position.x, camera->Position.y, camera->Position.z);
		context->set_last_render_position(last_render_position);

		HalfSphere *leftSide = new HalfSphere("../../assets/models/leftHalf.obj");
		HalfSphere *rightSide = new HalfSphere("../../assets/models/rightHalf.obj");
		Sphere *sphere = new Sphere(leftSide, rightSide);
		context->set_sphere(sphere);
		OptixModel *scene = loadOBJ(scene_file_path);
		context->set_optix_model(scene);
		RtAudio *dac = new RtAudio();
		uint32_t sample_rate;
		if (!audio_file_path.empty()) {
			try
			{
				AudioFile<float> *audio_file = new AudioFile<float>;
				context->set_audio_file_path(audio_file_path);
				audio_file->load(audio_file_path);
				context->set_audio_file(audio_file);
				sample_rate = audio_file->getSampleRate();
				size_t len_of_audio = audio_file->samples[0].size();
				size_t size_of_audio = sizeof(float) * len_of_audio;
				float *outputBuffer_left = (float *)malloc(size_of_audio);
				float *outputBuffer_right = (float *)malloc(size_of_audio);
				context->set_output_buffer_left(outputBuffer_left);
				context->set_output_buffer_right(outputBuffer_right);
				context->set_output_buffer_len(size_of_audio);
			}
			catch (const std::exception &)
			{
				return 1;
			}
		} else {
			sample_rate = 44100;
			context->set_live_flag(true);
		}
		
		context->set_sample_rate(sample_rate);
		placeReceiver(*sphere, scene, gdt::vec3f(camera->Position.x, camera->Position.y, camera->Position.z));
		AudioRenderer *renderer = new AudioRenderer(scene, ir_length_in_seconds, output_channels, sample_rate, materials);
		renderer->set_write_output_to_file_flag(write_output_to_file_on_render);
		renderer->set_write_ir_to_file_flag(write_ir_to_file_on_render);
		context->set_audio_renderer(renderer);
		context->set_re_render_distance_threshold(re_render_distance_threshold);

		thread screen1(screen);
		thread audio1(audio, dac, audio_file_path.empty()); // true = live input, false = audio

		screen1.join();
		audio1.detach();
		// Stop the stream
		RtAudioErrorType checkError = dac->stopStream();
		// if (dac.isStreamOpen())
		dac->closeStream();
		delete dac;
	}
	catch (const exception &e)
	{
		cerr << "Exception caught: " << e.what() << endl;
		return 1;
	}

	return 0;
}