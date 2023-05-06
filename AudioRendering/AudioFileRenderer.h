#pragma once

#include <vector>
#include <fstream>
#include <iomanip>

#include "AudioRenderingUtils.h"
#include "Camera.h"
#include "Scene.h"
#include "Source.h"

#include "AudioFile.h"

void renderAudioFile(
	Scene * scene, 
	glm::vec3 listener_pos, 
	float listener_size,
	glm::vec3 source_pos ,
	float source_power,
	const char * measurement_file_path,
	unsigned int measurement_length,
	int max_reflexions,
	float absorbtion_coef,
	int num_rays,
	timeInterval interval) {

	audioPaths * paths = new audioPaths();
	paths->ptr = NULL;
	paths->size = 0;
	paths->mutex = new std::mutex;

	RayTracer rt = RayTracer(scene, listener_pos, listener_size, source_pos, source_power, paths, max_reflexions, 1-absorbtion_coef, num_rays);

	rt.OmnidirectionalUniformSphereRayCast();

	//The size of Rs will depend on the lenght of the IR I want to mesure and the subdivision of that time length.
	//This means that if I want an IR to match the Rs used for auralization then I will have to simulate a 1 second IR
	//and multiply it by SAMPLE_RATE. 
	//However any other combination can be used depending on the desired outcome. If I just want to compare the result
	//of the simulation with the measurement, then the size will depend on the size and sample rate of the measurement file.
	
	AudioFile<float> measurement_file;
	measurement_file.load(measurement_file_path);

	auto sample_rate = measurement_file.getSampleRate();
	
	size_t size;
	if (measurement_length) {
		size = round(sample_rate * ((float)measurement_length/1000));
	}
	else {
		auto length = measurement_file.getLengthInSeconds();
		size = sample_rate * round(length);
	}

	std::vector<float> * rs = new std::vector<float>(size);

	//Initialize Rs
	std::fill(rs->begin(), rs->end(), 0.0);

	unsigned int interval_size = (interval.end - interval.begin) *  (sample_rate / 1000);
	std::vector<unsigned int> rays_in_interval(interval_size);
	std::fill(rays_in_interval.begin(), rays_in_interval.end(), 0);

	//Paths store the distance, to get the corresponding cell in vector Rs we need to find the elapsed time
	for (int i = 0; i < paths->size; i++) {
		float distance = paths->ptr[i].travelled_distance;
		float remaining_factor = paths->ptr[i].remaining_energy_factor;
		float elapsed_time = distance / SPEED_OF_SOUND;
		if (elapsed_time * 1000 > interval.begin && elapsed_time * 1000 < interval.end) {
			rays_in_interval[round((elapsed_time * 1000 - interval.begin) * (sample_rate / 1000))] += 1;
			//rays_in_interval.push_back(remaining_factor);
		}
		//The elapsed time is then converted to a position in the array by multiplying the time by the samples per second
		//This way a path that takes 1s to reach the listener will ocuppy the last position in the array.
		unsigned int array_pos = round(elapsed_time * sample_rate);
		if (array_pos < size && array_pos >= 0) {
			(*rs)[array_pos] += remaining_factor;
		}
	}
	std::ofstream rs_file("rs.txt");
	rs_file << std::setprecision(7);
	/*float received_energy = 0;*/
	for (int i = 0; i < size; i++) {
		rs_file << (*rs)[i] << ",";
		/*received_energy += (*rs)[i];*/
	}
	rs_file << std::endl;
	/*for (int i = 0; i < size; i++) {
		rs_file << measurement_file.samples[0][i] << ",";
	}
	rs_file << std::endl << received_energy;*/

	//rs_file << std::endl;
	//int direct_paths = 0;
	//for (int i = 0; i < paths->size; i++) {
	//	if (paths->ptr[i].is_direct_path) {
	//		direct_paths++;
	//		rs_file << paths->ptr[i].remaining_energy_factor << ",";
	//	}
	//}

	rs_file << std::endl;
	for (int i = 0; i < rays_in_interval.size(); i++) {
		rs_file << rays_in_interval.at(i) << ",";
	}

	rs_file.close();
}