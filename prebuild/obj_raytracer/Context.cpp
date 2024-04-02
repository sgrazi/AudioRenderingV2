#include "Context.h"

// Initialize the static instance pointer
Context *Context::instance = nullptr;

Context *Context::getInstance()
{
    if (!instance)
    {
        instance = new Context();
    }
    return instance;
}

void Context::showMessage()
{
    std::cout << "Singleton Context class instance created!" << std::endl;
}

void Context::set_audio_file(AudioFile<float>* audio_file)
{
    instance->audio_file = audio_file;
}

AudioFile<float>* Context::get_audio_file()
{
    return instance->audio_file;
}

void Context::set_ir_length_in_seconds(unsigned int ir_length_in_seconds)
{
    instance->ir_length_in_seconds = ir_length_in_seconds;
}

unsigned int Context::get_ir_length_in_seconds()
{
    return instance->ir_length_in_seconds;
}

void Context::set_output_channels(unsigned int output_channels)
{
    instance->output_channels = output_channels;
}

unsigned int Context::get_output_channels()
{
    return instance->output_channels;
}

void Context::set_volume(float volume)
{
    instance->volume = volume;
}

float Context::get_volume()
{
    return instance->volume;
}

void Context::set_sample_rate(uint32_t sample_rate)
{
    instance->sample_rate = sample_rate;
}

uint32_t Context::get_sample_rate()
{
    return instance->sample_rate;
}

void Context::set_output_buffer_left(float *output_buffer_left)
{
    instance->outputBuffer_left = output_buffer_left;
}

float *Context::get_output_buffer_left()
{
    return instance->outputBuffer_left;
}

void Context::set_output_buffer_right(float *output_buffer_right)
{
    instance->outputBuffer_right = output_buffer_right;
}

float *Context::get_output_buffer_right()
{
    return instance->outputBuffer_right;
}

void Context::set_output_buffer_len(size_t output_buffer_size)
{
    instance->output_buffer_size = output_buffer_size;
}

size_t Context::get_output_buffer_len()
{
    return instance->output_buffer_size;
}

void Context::set_scene_width(unsigned int width)
{
    instance->width = width;
}

unsigned int Context::get_scene_width()
{
    return instance->width;
}

void Context::set_scene_height(unsigned int height)
{
    instance->height = height;
}

unsigned int Context::get_scene_height()
{
    return instance->height;
}

void Context::set_scene_file_path(std::string scene_file_path)
{
    instance->scene_file_path = scene_file_path;
}

std::string Context::get_scene_file_path()
{
    return instance->scene_file_path;
}

void Context::set_audio_file_path(std::string audio_file_path)
{
    instance->audio_file_path = audio_file_path;
}

std::string Context::get_audio_file_path()
{
    return instance->audio_file_path;
}

void Context::set_material_file_path(std::string material_file_path)
{
    instance->material_file_path = material_file_path;
}

std::string Context::get_material_file_path()
{
    return instance->material_file_path;
}

void Context::set_sphere(Sphere *sphere)
{
    instance->sphere = sphere;
}

Sphere *Context::get_sphere()
{
    return instance->sphere;
}

void Context::set_optix_model(OptixModel *scene)
{
    instance->scene = scene;
}

OptixModel *Context::get_optix_model()
{
    return instance->scene;
}

Context::Context() {}

void Context::set_audio_renderer(AudioRenderer *renderer)
{
    instance->renderer = renderer;
}

AudioRenderer *Context::get_audio_renderer()
{
    return instance->renderer;
}

void Context::set_camera(Camera *camera)
{
    instance->camera = camera;
}

Camera *Context::get_camera()
{
    return instance->camera;
}

void Context::set_transmitter(std::vector<Mesh> *transmitterVector)
{
    instance->transmitterVector = transmitterVector;
}

std::vector<Mesh> *Context::get_transmitter()
{
    return instance->transmitterVector;
}

void Context::set_base_power(float base_power)
{
    instance->base_power = base_power;
}

float Context::get_base_power()
{
    return instance->base_power;
}

void Context::set_ray_distance_threshold(float ray_distance_threshold)
{
    instance->ray_distance_threshold = ray_distance_threshold;
}

float Context::get_ray_distance_threshold()
{
    return instance->ray_distance_threshold;
}

void Context::set_ray_energy_threshold(float ray_energy_threshold)
{
    instance->ray_energy_threshold = ray_energy_threshold;
}

float Context::get_ray_energy_threshold()
{
    return instance->ray_energy_threshold;
}

void Context::set_ray_max_bounces(unsigned int ray_max_bounces)
{
    instance->ray_max_bounces = ray_max_bounces;
}

unsigned int Context::get_ray_max_bounces()
{
    return instance->ray_max_bounces;
}

void Context::set_initial_emitter_pos(glm::vec3 initial_emitter_pos)
{
    instance->initial_emitter_pos = initial_emitter_pos;
}

glm::vec3 Context::get_initial_emitter_pos()
{
    return instance->initial_emitter_pos;
}

void Context::set_re_render_distance_threshold(float re_render_distance_threshold) {
    instance->re_render_distance_threshold = re_render_distance_threshold;
}

float Context::get_re_render_distance_threshold() {
    return instance->re_render_distance_threshold;
}

void Context::set_last_render_position(gdt::vec3f last_render_position) {
    instance->last_render_position = last_render_position;
}
gdt::vec3f Context::get_last_render_position() {
    return instance->last_render_position;
}

void Context::set_re_render_angle_threshold(float re_render_angle_threshold) {
    instance->re_render_angle_threshold = re_render_angle_threshold;
}

float Context::get_re_render_angle_threshold() {
    return instance->re_render_angle_threshold;
}
