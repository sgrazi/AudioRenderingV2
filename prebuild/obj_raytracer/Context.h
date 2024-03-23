#pragma once
#include <iostream>
#include "Sphere.h"
#include "Mesh.h"
#include "Camera.h"
#include "AudioRenderer.h"
#include "OptixModel.h"
#include <vector>
#include "AudioFile.h"

class Context
{
private:
	static Context *instance;
	Context();

	float volume;
	AudioFile<float>* audio_file;
	unsigned int ir_length_in_seconds = 2;
	unsigned int output_channels;
	unsigned int width;
	unsigned int height;
	float *outputBuffer_left;
	float *outputBuffer_right;

	size_t output_buffer_size;

	std::string scene_file_path;
	std::string audio_file_path;
	std::string material_file_path;

	float base_power;
	float ray_distance_threshold;
	float ray_energy_threshold;
	unsigned int ray_max_bounces;
	glm::vec3 initial_emitter_pos;

	uint32_t sample_rate;
	Sphere *sphere;
	std::vector<Mesh> *transmitterVector;
	OptixModel *scene;
	AudioRenderer *renderer;
	Camera *camera;
	float re_render_distance_threshold;
	gdt::vec3f last_render_position;

public:
	static Context *getInstance();
	void showMessage();

	// ------------------------------------ SOUND ------------------------------------

	static void set_audio_file(AudioFile<float>* audio_file);
	static AudioFile<float>* get_audio_file();

	static void set_ir_length_in_seconds(unsigned int ir_length_in_seconds);
	static unsigned int get_ir_length_in_seconds();

	static void set_output_channels(unsigned int output_channels);
	static unsigned int get_output_channels();

	static void set_volume(float volume);
	static float get_volume();

	static void set_sample_rate(uint32_t sample_rate);
	static uint32_t get_sample_rate();

	static void set_audio_file_path(std::string audio_file_path);
	static std::string get_audio_file_path();

	static void set_material_file_path(std::string material_file_path);
	static std::string get_material_file_path();

	static void set_base_power(float base_power);
	static float get_base_power();

	static void set_ray_distance_threshold(float ray_distance_threshold);
	static float get_ray_distance_threshold();

	static void set_ray_energy_threshold(float ray_energy_threshold);
	static float get_ray_energy_threshold();

	static void set_ray_max_bounces(uint32_t ray_max_bounces);
	static uint32_t get_ray_max_bounces();

	static void set_initial_emitter_pos(glm::vec3 initial_emitter_pos);
	static glm::vec3 get_initial_emitter_pos();

	static void set_output_buffer_left(float *output_buffer_left);
	static float *get_output_buffer_left();

	static void set_output_buffer_right(float *output_buffer_right);
	static float *get_output_buffer_right();

	static void set_output_buffer_len(size_t output_buffer_size);
	static size_t get_output_buffer_len();

	static void set_re_render_distance_threshold(float re_render_distance_threshold);
	static float get_re_render_distance_threshold();

	static void set_last_render_position(gdt::vec3f last_render_position);
	static gdt::vec3f get_last_render_position();

	// ------------------------------------ SCREEN ------------------------------------

	static void set_scene_width(unsigned int width);
	static unsigned int get_scene_width();

	static void set_scene_height(unsigned int height);
	static unsigned int get_scene_height();

	static void set_scene_file_path(std::string scene_file_path);
	static std::string get_scene_file_path();

	static void set_sphere(Sphere *sphere);
	static Sphere *get_sphere();

	static void set_optix_model(OptixModel *scene);
	static OptixModel *get_optix_model();

	static void set_audio_renderer(AudioRenderer *renderer);
	static AudioRenderer *get_audio_renderer();

	static void set_camera(Camera *camera);
	static Camera *get_camera();

	static void set_transmitter(std::vector<Mesh> *transmitterVector);
	static std::vector<Mesh> *get_transmitter();
};
