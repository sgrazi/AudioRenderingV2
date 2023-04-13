#pragma once

#include <mutex>
#include <glm/glm.hpp>

#include "Scene.h"
#include "Camera.h"
#include "Source.h"

#define LISTENER_SPHERE_RADIUS 2.0f
#define NUMBER_OF_RAYS 1000000
#define SOURCE_POWER 100000.0f
//Speed of sound in the air at 20 °C in m/s
#define SPEED_OF_SOUND 343
//Time that an audio sample takes in seconds
#define SAMPLE_RATE 16000
#define SAMPLE_DELTA_T 1 / SAMPLE_RATE
//#define SAMPLE_FORMAT RTAUDIO_SINT16
#define SAMPLE_FORMAT RTAUDIO_FLOAT32

//typedef signed short SAMPLE_TYPE;
typedef float SAMPLE_TYPE;

typedef struct rayHistory {
	float travelled_distance;
	float remaining_energy_factor;
	int reflection_num;
} rayHistory;

typedef struct audioPath {
	float travelled_distance;
	float remaining_energy_factor;
	bool is_direct_path;
} audioPath;

typedef struct audioPaths {
	audioPath * ptr;
	size_t size;
	//paths are used by audio library thread and main thread. (And possibly multiple rayCasting threads)
	std::mutex * mutex;
} audioPaths;

typedef struct intersectionData {
	float distance_to_sphere;
	float distance_inside_sphere;
};

typedef struct timeInterval {
	unsigned int begin;			//milliseconds
	unsigned int end;
} timeInterval;

class RayTracer {
public:
	Scene * scene;
	glm::vec3 listener_pos;
	float listener_size;
	glm::vec3 source_pos;
	float source_power;
	audioPaths * paths;
	int max_reflexions;
	float reflexion_coef;
	int num_rays;
public:
	RayTracer(Scene * scene,
		glm::vec3 listener_pos,
		float listener_size,
		glm::vec3 source_pos,
		float source_power,
		audioPaths * paths,
		int max_reflexions,
		float reflexion_coef,
		int num_rays);

	float rayIntensity(float remaining_energy, float distance_inside_sphere);

	//Returns the distance to the intersection if there is one, -1 if not.
	intersectionData raySphereIntersection(glm::vec3 origin, glm::vec3 dir, glm::vec3 center);

	/*
	 * Cast a single ray with origin (ox, oy, oz) and direction
	 * (dx, dy, dz).
	 */
	 //This function needs to do the intersection with the sound source and the reflection of the ray if it collides with geometry
	void castRay(
		glm::vec3 origin,
		glm::vec3 dir,
		rayHistory history);

	void OmnidirectionalUniformSphereRayCast();
	void OmnidirectionalHaltonSphereRayCast();

	void viewDirRayCast(Scene * scene, Camera * camera, Source * source);

	~RayTracer();
};
