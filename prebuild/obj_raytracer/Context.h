#pragma once
#include <iostream>
#include "Sphere.h"
#include "Mesh.h"
#include "Camera.h"
#include "AudioRenderer.h"
#include "OptixModel.h"
#include <vector>

class Context {
private:
    static Context* instance;
    Context();
	
	unsigned int ir_length_in_seconds = 2; 
	unsigned int output_channels;
	unsigned int width;
	unsigned int height;
	float volume;
	std::string file_path;
	uint32_t sample_rate;
	Sphere* sphere;
	std::vector<Mesh>* transmitterVector;
	OptixModel* scene;
	AudioRenderer* renderer;
	Camera* camera;
	float* outputBuffer;


public:
    static Context* getInstance();
    void showMessage();

	// ------------------------------------ SOUND ------------------------------------

	static void set_ir_length_in_seconds(unsigned int ir_length_in_seconds);
	static unsigned int get_ir_length_in_seconds();

	static void set_output_channels(unsigned int output_channels);
	static unsigned int get_output_channels();

	static void set_volume(float volume);
	static float get_volume();

	static void set_sample_rate(uint32_t sample_rate);
	static uint32_t get_sample_rate();

	static void set_output_buffer(float* output_buffer);
	static float* get_output_buffer();

	// ------------------------------------ SCREEN ------------------------------------

	static void set_scene_width(unsigned int width);
	static unsigned int get_scene_width();

	static void set_scene_height(unsigned int height);
	static unsigned int get_scene_height();

	static void set_file_path(std::string file_path);
	static std::string get_file_path();

	static void set_sphere(Sphere* sphere);
	static Sphere* get_sphere();

	static void set_optix_model(OptixModel* scene);
	static OptixModel* get_optix_model();

	static void set_audio_renderer(AudioRenderer* renderer);
	static AudioRenderer* get_audio_renderer();

	static void set_camera(Camera* camera);
	static Camera* get_camera();

	static void set_transmitter(std::vector<Mesh>* transmitterVector);
	static std::vector<Mesh>* get_transmitter();
};

