#pragma once
#ifndef UTILS_H
#define UTILS_H
#define SAMPLE_TYPE double

#include <glm/glm.hpp>
#include <vector>
#include <mutex>
#include "gdt/math/AffineSpace.h"
#include "AudioFile.h"
#include "CircularBuffer.h"
#include "Experimentation.h"

struct audioPaths
{
	void* ptr;
	int size;
};

struct audioCallbackData
{
	int bufferFrames;
	int pos;
	int samplesRecordBufferSize;
	CircularBuffer<SAMPLE_TYPE>* samplesRecordBuffer;
	audioPaths* paths;
	float volume;
	std::mutex* inputBufferMutex;
};

struct AudioInfo
{
	AudioFile<float>* audio;
	float* volumen;
};

glm::vec3 gdt2glm(gdt::vec3f vector);

float distanceP2P(gdt::vec3f p1, gdt::vec3f p2);

double median(std::vector<double> values);

void process_files_with_prefix(const std::string& directoryPath, const std::string& prefix);

void process_file(const std::string& filePath);
#endif
