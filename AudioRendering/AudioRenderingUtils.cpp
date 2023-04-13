#include "AudioRenderingUtils.h"

#include<random>
#include<cmath>
#include<chrono>
#include <vector>
#include "Halton.h"
#include "halton_sampler.h"
#include <ctime>
#include <iostream>

RayTracer::RayTracer(Scene * scene,
	glm::vec3 listener_pos,
	float listener_size,
	glm::vec3 source_pos,
	float source_power,
	audioPaths * paths,
	int max_reflexions,
	float reflexion_coef,
	int num_rays) {
	this->scene = scene;
	this->listener_pos = listener_pos;
	this->listener_size = listener_size;
	this->source_pos = source_pos;
	this->source_power = source_power;
	this->paths = paths;
	this->max_reflexions = max_reflexions;
	this->reflexion_coef = reflexion_coef;
	this->num_rays = num_rays;
}

intersectionData RayTracer::raySphereIntersection(glm::vec3 origin, glm::vec3 dir, glm::vec3 center) {
	//The following is obtained from solving the ecuation system given by the ray and sphere
	//The result is a second degree ecuation: at^2 + bt + c = 0
	float a = glm::dot(dir, dir);
	float b = 2 * glm::dot(dir, origin - center);
	float c = glm::dot(origin - center, origin - center) - pow(this->listener_size, 2);
	float discriminant = (pow(b, 2) - 4 * a*c);
	if (discriminant < 0) {
		return { -1, 0 };
	}
	else {
		float t1 = (-b - sqrt(discriminant)) / (2.0*a);
		float t2 = (-b + sqrt(discriminant)) / (2.0*a);
		if (t1 > 0 && t2 > 0) {
			if (t2 > t1) {
				return { t1, t2 - t1 };
			}else {
				return { t2, t1 - t2 };
			}
		}
		return { -1, 0 };
	}
}

float RayTracer::rayIntensity(float remaining_energy, float distance_inside_sphere) {
	return distance_inside_sphere * remaining_energy / ((4 / 3) * M_PI * pow(this->listener_size, 3));
}

/*
 * Cast a single ray with origin (ox, oy, oz) and direction
 * (dx, dy, dz).
 */
 //This function needs to do the intersection with the sound source and the reflection of the ray if it collides with geometry
void RayTracer::castRay(
	glm::vec3 origin,
	glm::vec3 dir,
	rayHistory history)
{
	/*
	 * The intersect context can be used to set intersection
	 * filters or flags, and it also contains the instance ID stack
	 * used in multi-level instancing.
	 */
	struct RTCIntersectContext context;
	rtcInitIntersectContext(&context);
	/*
	 * The ray hit structure holds both the ray and the hit.
	 * The user must initialize it properly -- see API documentation
	 * for rtcIntersect1() for details.
	 */
	struct RTCRayHit rayhit;
	rayhit.ray.org_x = origin.x;
	rayhit.ray.org_y = origin.y;
	rayhit.ray.org_z = origin.z;
	rayhit.ray.dir_x = dir.x;
	rayhit.ray.dir_y = dir.y;
	rayhit.ray.dir_z = dir.z;
	rayhit.ray.tnear = 0;
	rayhit.ray.tfar = std::numeric_limits<float>::infinity();
	rayhit.ray.mask = -1;
	rayhit.ray.flags = 0;
	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

	rtcIntersect1(this->scene->getRTCScene(), &context, &rayhit);

	//Check if ray interescts listener
	intersectionData intersection_data = raySphereIntersection(origin, dir, listener_pos);
	//Check if ray intersects room
	if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
	{
		//If listener intersection found
		if (intersection_data.distance_to_sphere >= 0) {
			//If listener intersection is before room interection
			if (intersection_data.distance_to_sphere < rayhit.ray.tfar) {
				//Add distance_to_source to overall distance
				//Calculate parameters for transfer function (e.g. absorption from specular reflections)
				//Return parameters and traveled distance to add to the histogram
				//printf("Found intersection with listener. %i\n", history.reflection_num);
				//Add path to paths
				audioPath newAudioPath = { history.travelled_distance + intersection_data.distance_to_sphere, rayIntensity(history.remaining_energy_factor, intersection_data.distance_inside_sphere), history.reflection_num == 0 };
				this->paths->mutex->lock();
				this->paths->size++;
				this->paths->ptr = (audioPath*)realloc(paths->ptr, paths->size * sizeof(audioPath));
				this->paths->ptr[paths->size - 1] = newAudioPath;
				this->paths->mutex->unlock();
				return;
			}
			else {
				//printf("Intersection blocked by geometry. %i\n", history.reflection_num);
			}
		}

		//Calculate remaining energy if less than something also return
		if (history.reflection_num > this->max_reflexions) {
			//printf("Ray exahusted.\n");
			return;
		}
		//Reflect ray with geometry normal
		glm::vec3 new_dir;
		glm::vec3 normal = glm::normalize(glm::vec3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z));
		if (glm::dot(dir, normal) < 0) {
			new_dir = glm::reflect(dir, normal);
		}
		else {
			new_dir = glm::reflect(dir, -normal);
		}
		//New origin is obtained by moving tfar in the ray direction from the current origin
		glm::vec3 new_origin = origin + dir * rayhit.ray.tfar;
		//When casting new ray new origin must me moved delta in the new direction to avoid numeric errors. (Ray begining inside the geometry)
		history.reflection_num++;
		history.remaining_energy_factor *= reflexion_coef;
		history.travelled_distance += rayhit.ray.tfar;
		//if (history.remaining_energy_factor > 0.0000000001) {
			castRay(new_origin + new_dir * 0.01f, new_dir, history);
		//}

		/* Note how geomID and primID identify the geometry we just hit.
		 * We could use them here to interpolate geometry information,
		 * compute shading, etc. */
	}
	else {
		if (intersection_data.distance_to_sphere >= 0) {
			//Add distance_to_source to overall distance
			//Calculate parameters for transfer function (e.g. absorption from specular reflections)
			//Return parameters and traveled distance to add to the histogram
			//printf("Found intersection with listener. %i\n", history.reflection_num);
			audioPath newAudioPath = { history.travelled_distance + intersection_data.distance_to_sphere, rayIntensity(history.remaining_energy_factor, intersection_data.distance_inside_sphere), history.reflection_num == 0 };
			this->paths->mutex->lock();
			this->paths->size++;
			this->paths->ptr = (audioPath*)realloc(paths->ptr, paths->size * sizeof(audioPath));
			this->paths->ptr[paths->size - 1] = newAudioPath;
			this->paths->mutex->unlock();
			return;
		}
	}
	//printf("No intersection with listener found.\n");
	return;
}

void RayTracer::OmnidirectionalUniformSphereRayCast()
{
	//If we are rendering audio again then we celar previously found paths
	if (this->paths->ptr) {
		free(this->paths->ptr);
		this->paths->ptr = NULL;
		this->paths->size = 0;
	}

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);
	std::uniform_real_distribution<double> uniform01(0.0, 1.0);

	for (int i = 0; i < this->num_rays; ++i) {
		double theta = 2 * M_PI * uniform01(generator);
		double phi = acos(1 - 2 * uniform01(generator));
		double dx = sin(phi) * cos(theta);
		double dy = sin(phi) * sin(theta);
		double dz = cos(phi);
		glm::vec3 dir = glm::normalize(glm::vec3(dx, dy, dz));
		rayHistory new_ray_history = { 0.0f, this->source_power / this->num_rays, 0 };
		castRay(source_pos, dir, new_ray_history);
	}

	////srand(time(NULL));
	//float rnd1 = uniform01(generator);
	//float rnd2 = uniform01(generator);

	//int n = 20;		//The number of subdivisions in the sphere is n^2
	//for (int i = 0; i < n; ++i) {
	//	for (int j = 0; j < n; ++j) {
	//		float dx = 2 * sqrtf((i + rnd1) / n - pow((i + rnd1) / n, 2)) * cos(2 * M_PI*(j + rnd2) / n);
	//		float dy = 2 * sqrtf((i + rnd1) / n - pow((i + rnd1) / n, 2)) * sin(2 * M_PI*(j + rnd2) / n);
	//		float dz = 1 - 2 * (i + rnd1) / n;
	//		glm::vec3 dir = glm::normalize(glm::vec3(dx, dy, dz));
	//		castRay(scene->getRTCScene(), source->pos, dir, camera->pos);
	//	}
	//}
}

void RayTracer::OmnidirectionalHaltonSphereRayCast()
{
	//If we are rendering audio again then we celar previously found paths
	if (this->paths->ptr) {
		free(this->paths->ptr);
		this->paths->ptr = NULL;
		this->paths->size = 0;
	}

	//Halton_sampler sampler = Halton_sampler();
	//sampler.init_faure();

	for (int i = 0; i < this->num_rays; ++i) {
		double* phitheta = halton(i, 2);
		/*float halton_x = sampler.sample(2, i);
		float halton_y = sampler.sample(3, i);*/
		double theta = 2 * M_PI * phitheta[1];
		double phi = acos(1 - 2 * phitheta[0]);
		delete[] phitheta;
		double dx = sin(phi) * cos(theta);
		double dy = sin(phi) * sin(theta);
		double dz = cos(phi);
		glm::vec3 dir = glm::normalize(glm::vec3(dx, dy, dz));
		rayHistory new_ray_history = { 0.0f, this->source_power / this->num_rays, 0 };
		castRay(source_pos, dir, new_ray_history);
	}
}

void RayTracer::viewDirRayCast(Scene * scene, Camera * camera, Source * source) {
	rayHistory new_ray_history = { 0.0f, 1.0f, 0 };
	castRay(camera->pos, camera->ref - camera->pos, new_ray_history);
}

RayTracer::~RayTracer() {

}