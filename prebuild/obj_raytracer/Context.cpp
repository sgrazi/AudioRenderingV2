#include "Context.h"

// Initialize the static instance pointer
Context* Context::instance = nullptr;

Context* Context::getInstance() {
    if (!instance) {
        instance = new Context();
    }
    return instance;
}

void Context::showMessage() {
    std::cout << "Singleton Context class instance created!" << std::endl;
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

void Context::set_volume(float volume) {
    instance->volume = volume;
}

float Context::get_volume() {
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

void Context::set_output_buffer(float* output_buffer)
{
    instance->outputBuffer = output_buffer;
}

float* Context::get_output_buffer()
{
    return instance->outputBuffer;
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

void Context::set_file_path(std::string file_path)
{
    instance->file_path = file_path;
}

std::string Context::get_file_path()
{
    return instance->file_path;
}

void Context::set_sphere(Sphere* sphere)
{
    instance->sphere = sphere;
}

Sphere* Context::get_sphere()
{
    return instance->sphere;
}

void Context::set_optix_model(OptixModel* scene)
{
    instance->scene = scene;
}

OptixModel* Context::get_optix_model()
{
    return instance->scene;
}

Context::Context() {}

void Context::set_audio_renderer(AudioRenderer* renderer)
{
    instance->renderer = renderer;
}

AudioRenderer* Context::get_audio_renderer()
{
    return instance->renderer;
}

void Context::set_camera(Camera* camera)
{
    instance->camera = camera;
}

Camera* Context::get_camera()
{
    return instance->camera;
}

void Context::set_transmitter(std::vector<Mesh>* transmitterVector)
{
    instance->transmitterVector = transmitterVector;
}

std::vector<Mesh>* Context::get_transmitter()
{
    return instance->transmitterVector;
}
