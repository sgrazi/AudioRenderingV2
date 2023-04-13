#pragma once
/*Audio samples contain values of amplitude (loudness). If we have a sample rate of 44100Hz, 
frequency is derived from the variations in these amplitudes considering that two 
contiguous samples are 1/44100 seconds apart.*/

#include "RtAudio.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_common.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "Scene.h"
#include "Camera.h"
#include "Source.h"
#include "CircularBuffer.h"
//#include "thread_pool.hpp"

#include<random>
#include<cmath>
#include<chrono>

#include "AudioRenderingUtils.h"
#include "AudioFile.h"

typedef struct audioCallbackData {
	unsigned int bufferFrames;
	unsigned int pos;
	/*A buffer to store old samples to use in posterior calculations.
	To store 1 second of samples size should be: n = sampleRate, to store 2 seconds n = sampleRate x 2*/
	unsigned int samplesRecordBufferSize;
	CircularBuffer<SAMPLE_TYPE> * samplesRecordBuffer;
	audioPaths * paths;
	/*Rs's type needs to be compatible with SAMPLE_TYPE.
	If Rs's values go from 0 to 1, then the SAMPLE_TYPE should allow decimal values*/
	std::vector<float> * Rs;
	//thread_pool * pool;
	float volume;
} audioCallbackData;

typedef struct streamParameters {
	RtAudio::StreamParameters *iParams, *oParams;
	unsigned int * bufferFrames;
	RtAudio::StreamOptions * options;
} streamParameters;

class AudioRenderer {
public:
	//buffer to store all frames up to n seconds
	RtAudio * audioApi;
	audioPaths * currentPaths; //paths result of the audio render
	streamParameters * streamParams;
	unsigned int * bufferBytes;
	audioCallbackData * audioData;

	//Simulation data
	int max_reflexions;
	float absorbtion_coef;
	int num_rays;
	float source_power;
	float listener_size;
	int sample_rate;
	AudioFile<float> audio_sample_file;

public:
	AudioRenderer(){};
	AudioRenderer(int max_reflexions, float absorbtion_coef, int num_rays, float source_power, float listener_size, int sample_rate);
	AudioRenderer(int max_reflexions, float absorbtion_coef, int num_rays, float source_power, float listener_size, int sample_rate, const char * audio_sample);
	void resetStream();
	void render(Scene * scene, Camera * camera, Source * source);
	void updateVolume(float value);
	~AudioRenderer();
};