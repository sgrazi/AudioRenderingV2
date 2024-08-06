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
#endif
