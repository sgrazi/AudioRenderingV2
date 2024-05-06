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

bool Context::loadContext(cJSON *config)
{
	// renderer_parameters
	const cJSON *cJSON_renderer_parameters = cJSON_GetObjectItem(config, "renderer_parameters");
	// defaults
	float initial_volume = 1.0f;
	unsigned int ir_length_in_seconds = 2;
	unsigned int width = 1366;
	unsigned int height = 768;
	bool write_ir_to_file_on_render = false;
	bool write_output_to_file_on_render = false;
	float re_render_distance_threshold = 3;
	float re_render_angle_threshold = 5;
	if (cJSON_IsObject(cJSON_renderer_parameters))
	{
		cJSON *cJSON_initial_volume = cJSON_GetObjectItem(cJSON_renderer_parameters, "initial_volume");
		if (cJSON_IsNumber(cJSON_initial_volume))
			initial_volume = cJSON_initial_volume->valuedouble;

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

		const cJSON *cJSON_re_render_angle_threshold = cJSON_GetObjectItem(cJSON_renderer_parameters, "re_render_angle_threshold");
		if (cJSON_IsBool(cJSON_re_render_angle_threshold))
			re_render_angle_threshold = cJSON_IsTrue(cJSON_re_render_angle_threshold);
	}
	// scene_parameters
	const cJSON *cJSON_scene_parameters = cJSON_GetObjectItem(config, "scene_parameters");
	// defaults
	std::string scene_file_path = "../../assets/models/1D_U.obj";
	std::string audio_file_path;
	std::string materials_file_path = "";
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
	std::vector<Material> materials;
	if (cJSON_IsObject(cJSON_pathtracer_parameters))
	{
		cJSON *cJSON_base_power = cJSON_GetObjectItem(cJSON_pathtracer_parameters, "base_power");
		if (cJSON_IsNumber(cJSON_base_power))
			base_power = cJSON_base_power->valuedouble;

		const cJSON *cJSON_rays_size = cJSON_GetObjectItem(cJSON_pathtracer_parameters, "rays");
		if (cJSON_IsObject(cJSON_rays_size))
		{
			cJSON *x = cJSON_GetObjectItem(cJSON_rays_size, "x");
			cJSON *y = cJSON_GetObjectItem(cJSON_rays_size, "y");
			cJSON *z = cJSON_GetObjectItem(cJSON_rays_size, "z");
			if (cJSON_IsNumber(x) && cJSON_IsNumber(y) && cJSON_IsNumber(z))
				rays = glm::vec3(x->valuedouble, y->valuedouble, z->valuedouble);
		}

		cJSON *cJSON_ray_distance_threshold = cJSON_GetObjectItem(cJSON_pathtracer_parameters, "ray_distance_threshold");
		if (cJSON_IsNumber(cJSON_ray_distance_threshold))
			ray_distance_threshold = cJSON_ray_distance_threshold->valueint;

		cJSON *cJSON_ray_energy_threshold = cJSON_GetObjectItem(cJSON_pathtracer_parameters, "ray_energy_threshold");
		if (cJSON_IsNumber(cJSON_ray_energy_threshold))
			ray_energy_threshold = cJSON_ray_energy_threshold->valuedouble;

		const cJSON *cJSON_ray_max_bounces = cJSON_GetObjectItem(cJSON_pathtracer_parameters, "ray_max_bounces");
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

	Context *context = Context::getInstance();
	context->set_volume(initial_volume);
	context->set_ir_length_in_seconds(ir_length_in_seconds);
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
	std::vector<Mesh> *transmitterVector = new std::vector<Mesh>;
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

	size_t sample_rate;
	if (!audio_file_path.empty()) {
		try
		{
			AudioFile<float>* audio_file = new AudioFile<float>;
			context->set_audio_file_path(audio_file_path);
			audio_file->load(audio_file_path);
			context->set_audio_file(audio_file);
			sample_rate = audio_file->getSampleRate();
			size_t len_of_audio = audio_file->samples[0].size();
			size_t size_of_audio = sizeof(float) * len_of_audio;
			float* outputBuffer_left = (float*)malloc(size_of_audio);
			float* outputBuffer_right = (float*)malloc(size_of_audio);
			context->set_output_buffer_left(outputBuffer_left);
			context->set_output_buffer_right(outputBuffer_right);
			context->set_output_buffer_len(size_of_audio);
		}
		catch (const std::exception& e)
		{
			std::cerr << "Exception caught: " << e.what() << std::endl;
			return false;
		}
	}
	else {
		sample_rate = 44100;
		context->set_live_flag(true);
	}
	context->set_sample_rate(sample_rate);

	placeReceiver(*sphere, scene, gdt::vec3f(camera->Position.x, camera->Position.y, camera->Position.z), camera->globalAngle);

	AudioRenderer *renderer = new AudioRenderer(scene, ir_length_in_seconds, sample_rate, materials);
	renderer->set_write_output_to_file_flag(write_output_to_file_on_render);
	renderer->set_write_ir_to_file_flag(write_ir_to_file_on_render);
	context->set_audio_renderer(renderer);
	context->set_re_render_distance_threshold(re_render_distance_threshold);
	context->set_re_render_angle_threshold(re_render_angle_threshold);
}

void Context::set_audio_file(AudioFile<float> *audio_file)
{
	instance->audio_file = audio_file;
}

AudioFile<float> *Context::get_audio_file()
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

void Context::set_re_render_distance_threshold(float re_render_distance_threshold)
{
	instance->re_render_distance_threshold = re_render_distance_threshold;
}

float Context::get_re_render_distance_threshold()
{
	return instance->re_render_distance_threshold;
}

void Context::set_last_render_position(gdt::vec3f last_render_position)
{
	instance->last_render_position = last_render_position;
}
gdt::vec3f Context::get_last_render_position()
{
	return instance->last_render_position;
}

void Context::set_live_input_buffer(CircularBuffer<double> *b)
{
	instance->liveInputBuffer = b;
}

CircularBuffer<double> *Context::get_live_input_buffer()
{
	return instance->liveInputBuffer;
}

void Context::set_live_flag(bool flag)
{
	instance->liveFlag = flag;
}

bool Context::get_live_flag()
{
	return instance->liveFlag;
}

void Context::set_re_render_angle_threshold(float re_render_angle_threshold)
{
	instance->re_render_angle_threshold = re_render_angle_threshold;
}

float Context::get_re_render_angle_threshold()
{
	return instance->re_render_angle_threshold;
}

void Context::set_is_rendering(bool is_rendering)
{
	instance->is_rendering = is_rendering;
}

bool Context::get_is_rendering()
{
	return instance->is_rendering;
}
