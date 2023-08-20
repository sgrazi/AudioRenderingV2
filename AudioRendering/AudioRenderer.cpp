#include "AudioRenderer.h"
#include <mutex>
#include <thread>
#include <fstream>
#include <iomanip>

std::mutex outputBufferMutex;
std::mutex inputBufferMutex;

//void rowColumnProduct(int begin, int end, void * outputBuffer, audioCallbackData * renderData) {
//	for (int i = begin; i <= end; i++) {
//		SAMPLE_TYPE output_value = 0;
//		for (int j = 0; j < renderData->samplesRecordBufferSize - i; j++) {
//			output_value += (*renderData->Rs)[j] * renderData->samplesRecordBuffer->getElement(renderData->samplesRecordBufferSize - 1 - i - j);
//		}
//		outputBufferMutex.lock();
//		((SAMPLE_TYPE*)outputBuffer)[renderData->bufferFrames - 1 - i] = output_value;
//		outputBufferMutex.unlock();
//	}
//}

//Callback that processes audio frames
int processAudio(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
	double streamTime, RtAudioStreamStatus status, void *data)
{
	if (status) std::cout << "Stream over/underflow detected." << std::endl;

	audioCallbackData *renderData = (audioCallbackData *)data;
	memset(outputBuffer, 0, renderData->bufferFrames * sizeof(SAMPLE_TYPE));
	//First we copy the new samples into our buffer
	renderData->samplesRecordBuffer->insert((SAMPLE_TYPE*)inputBuffer, renderData->bufferFrames);
	if (renderData->samplesRecordBuffer->full) {
		//Then we process the frames
		/*int frames_per_thread = round(renderData->bufferFrames / 4);
		renderData->pool->push_task(rowColumnProduct, 0, frames_per_thread - 1, outputBuffer, renderData);
		renderData->pool->push_task(rowColumnProduct, frames_per_thread, frames_per_thread * 2 - 1, outputBuffer, renderData);
		renderData->pool->push_task(rowColumnProduct, frames_per_thread * 2, frames_per_thread * 3 - 1, outputBuffer, renderData);
		renderData->pool->push_task(rowColumnProduct, frames_per_thread * 3, renderData->bufferFrames - 1, outputBuffer, renderData);
		
		renderData->pool->wait_for_tasks();*/

		unsigned int RvIndex;

		for (int i = 0; i < renderData->bufferFrames; i++) {
			SAMPLE_TYPE output_value = 0;
			for (int j = 0; j < renderData->samplesRecordBufferSize - i; j++) {
				output_value += (*renderData->Rs)[j] * renderData->samplesRecordBuffer->getElement(renderData->samplesRecordBufferSize - 1 - i - j);
			}
			outputBufferMutex.lock();
			//Output has 2 channels
			RvIndex = (renderData->bufferFrames * 2) - 1 - (i * 2);
			((SAMPLE_TYPE*)outputBuffer)[RvIndex] = output_value * renderData->volume;
			((SAMPLE_TYPE*)outputBuffer)[RvIndex - 1] = output_value * renderData->volume;
			outputBufferMutex.unlock();
		}

		//For every element in Rs
		//for (int i = 0; i < renderData->samplesRecordBufferSize; i++) {
		//	//for each element in rho's column
		//	for (int j = 0; j < renderData->bufferFrames && j < renderData->samplesRecordBufferSize - i; j++) {
		//		float value = (*renderData->Rs)[i] * renderData->samplesRecordBuffer->getElement(renderData->samplesRecordBufferSize - 1 - i - j);
		//		RvIndex = (renderData->bufferFrames * 2) - 1 - (j * 2);
		//		((SAMPLE_TYPE*)outputBuffer)[RvIndex] += value;
		//		((SAMPLE_TYPE*)outputBuffer)[RvIndex - 1] += value;
		//	}
		//	
		//}
	}
	return 0;
}

//Callback that processes audio frames
int processAudioSample(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
	double streamTime, RtAudioStreamStatus status, void *data)
{
	if (status) std::cout << "Stream over/underflow detected." << std::endl;

	audioCallbackData *renderData = (audioCallbackData *)data;
	memset(outputBuffer, 0, renderData->bufferFrames * sizeof(SAMPLE_TYPE));

	unsigned int RvIndex;

	// To have a window of 1 second of samples is to have a windoe of size equal to se samplerate. A 2 second window is (2 * sample_rate) long.
	// This is implicit in Rs's size. We compute it so the window is 1 second long.
	size_t size = renderData->Rs->size();

	for (int i = 0; i < renderData->bufferFrames; i++) {
		SAMPLE_TYPE output_value = 0;
		for (int j = 0; j < size - i; j++) {
			output_value += (*renderData->Rs)[j] * renderData->samplesRecordBuffer->getElement(size - 1 - i - j);
		}
		outputBufferMutex.lock();
		//Output has 2 channels
		RvIndex = (renderData->bufferFrames * 2) - 1 - (i * 2);
		((SAMPLE_TYPE*)outputBuffer)[RvIndex] = output_value  * renderData->volume;
		((SAMPLE_TYPE*)outputBuffer)[RvIndex - 1] = output_value * renderData->volume;
		outputBufferMutex.unlock();
	}
	for (int i = 0; i < renderData->bufferFrames; i++) {
		renderData->samplesRecordBuffer->setElement(i, 0);
	}

	//for (int i = 0; i < renderData->bufferFrames; i++) {
	//	outputBufferMutex.lock();
	//	//Output has 2 channels
	//	((SAMPLE_TYPE*)outputBuffer)[i*2] = renderData->samplesRecordBuffer->getElement(i);
	//	((SAMPLE_TYPE*)outputBuffer)[(i*2)+1] = renderData->samplesRecordBuffer->getElement(i);
	//	outputBufferMutex.unlock();
	//}

	//for (int i = 0; i < renderData->bufferFrames; i++) {
	//	renderData->samplesRecordBuffer->setElement(i, 0);
	//}

	renderData->samplesRecordBuffer->head = (renderData->samplesRecordBuffer->head + renderData->bufferFrames) % renderData->samplesRecordBufferSize;
	renderData->samplesRecordBuffer->tail = (renderData->samplesRecordBuffer->tail + renderData->bufferFrames) % renderData->samplesRecordBufferSize;

	return 0;
}

int inout(void *outputBuffer, void *inputBuffer, unsigned int /*nBufferFrames*/,
	double /*streamTime*/, RtAudioStreamStatus status, void *data)
{
	// Since the number of input and output channels is equal, we can do
	// a simple buffer copy operation here.
	if (status) std::cout << "Stream over/underflow detected." << std::endl;

	memcpy(outputBuffer, inputBuffer, 512);
	return 0;
}

AudioRenderer::AudioRenderer(int max_reflexions, float absorbtion_coef, int num_rays, float source_power, float listener_size, int sample_rate) {
	this->max_reflexions = max_reflexions;
	this->absorbtion_coef = absorbtion_coef;
	this->num_rays = num_rays;
	this->source_power = source_power;
	this->listener_size = listener_size;
	this->sample_rate = sample_rate;

	//Init audio stream
	this->audioApi = new RtAudio();

	if (this->audioApi->getDeviceCount() < 1) {
		std::cout << "\nNo audio devices found!\n";
		exit(0);
	}

	unsigned int bufferBytes, bufferFrames = 512, input_channles = 1, output_channels = 2;

	//Set up stream parameters they need to be in heap since audio api will use them in separate thread
	this->streamParams = new streamParameters();
	this->streamParams->iParams = new RtAudio::StreamParameters();
	this->streamParams->iParams->deviceId = this->audioApi->getDefaultInputDevice();
	this->streamParams->iParams->nChannels = input_channles;
	this->streamParams->iParams->firstChannel = 0;
	this->streamParams->oParams = new RtAudio::StreamParameters();
	this->streamParams->oParams->deviceId = this->audioApi->getDefaultOutputDevice();
	this->streamParams->oParams->nChannels = output_channels;
	this->streamParams->oParams->firstChannel = 0;
	this->streamParams->bufferFrames = new unsigned int(bufferFrames);
	this->streamParams->options = new RtAudio::StreamOptions();

	this->bufferBytes = new unsigned int(bufferFrames * input_channles * sizeof(SAMPLE_TYPE));

	//This whole struct will be accessed from the audio api thread, so it needs to be in heap.
	this->audioData = new audioCallbackData();
	//Total frames in a buffer for all channels
	this->audioData->bufferFrames = bufferFrames * input_channles;
	this->audioData->pos = 0;
	//1 second's worth of samples.
	this->audioData->samplesRecordBufferSize = this->sample_rate * input_channles;
	this->audioData->samplesRecordBuffer = new CircularBuffer<SAMPLE_TYPE>(this->audioData->samplesRecordBufferSize);
	this->audioData->paths = new audioPaths();
	this->audioData->paths->ptr = NULL;
	this->audioData->paths->size = 0;
	this->audioData->paths->mutex = new std::mutex();
	this->audioData->volume = 30.0f;
	//this->audioData->pool = new thread_pool(4);

	try {
		this->audioApi->openStream(this->streamParams->oParams, this->streamParams->iParams, SAMPLE_FORMAT, this->sample_rate,
			this->streamParams->bufferFrames, &processAudio, (void *)this->audioData, this->streamParams->options);
	}
	catch (RtAudioError& e) {
		e.printMessage();
		exit(0);
	}

	try {
		this->audioApi->startStream();
	}
	catch (RtAudioError& e) {
		e.printMessage();
		exit(0);
	}

	this->currentPaths = new audioPaths();
	this->currentPaths->ptr = NULL;
	this->currentPaths->size = 0;
	this->currentPaths->mutex = new std::mutex;

	//Create Rs vector
	this->audioData->Rs = new std::vector<float>(this->audioData->samplesRecordBufferSize);
}

AudioRenderer::AudioRenderer(int max_reflexions, float absorbtion_coef, int num_rays, float source_power, float listener_size, int sample_rate, const char* audio_sample) {
	//Same as other constructor but audiostream is not started. Instead, stream is (created and) started on demand
	this->max_reflexions = max_reflexions;
	this->absorbtion_coef = absorbtion_coef;
	this->num_rays = num_rays;
	this->source_power = source_power;
	this->listener_size = listener_size;
	this->sample_rate = 256;

	//Init audio stream
	this->audioApi = new RtAudio();

	if (this->audioApi->getDeviceCount() < 1) {
		std::cout << "\nNo audio devices found!\n";
		exit(0);
	}

	unsigned int bufferBytes, bufferFrames = 256, input_channles = 1, output_channels = 2;

	//Set up stream parameters they need to be in heap since audio api will use them in separate thread
	this->streamParams = new streamParameters();
	this->streamParams->iParams = new RtAudio::StreamParameters();
	this->streamParams->iParams->deviceId = this->audioApi->getDefaultInputDevice();
	this->streamParams->iParams->nChannels = input_channles;
	this->streamParams->iParams->firstChannel = 0;
	this->streamParams->oParams = new RtAudio::StreamParameters();
	this->streamParams->oParams->deviceId = this->audioApi->getDefaultOutputDevice();
	this->streamParams->oParams->nChannels = output_channels;
	this->streamParams->oParams->firstChannel = 0;
	this->streamParams->bufferFrames = new unsigned int(bufferFrames);
	this->streamParams->options = new RtAudio::StreamOptions();

	//This whole struct will be accessed from the audio api thread, so it needs to be in heap.
	this->audioData = new audioCallbackData();
	//Total frames in a buffer for all channels
	this->audioData->bufferFrames = bufferFrames;
	this->audioData->pos = 0;
	this->audio_sample_file.load(audio_sample);

	// The size of the buffer depends on the lenght of the audio file. We want to load the samples just once, so we don't have to do any extra computations while the sample is playing.
	// The buffer needs to fit the entire audio sample, plus, an empty space of size sample_rate so the stream can start just before the audio starts playing.
	
	int audio_length = ceil(this->audio_sample_file.getLengthInSeconds());
	this->audioData->samplesRecordBufferSize = (audio_length + 1) * this->sample_rate;
	this->audioData->samplesRecordBuffer = new CircularBuffer<SAMPLE_TYPE>(this->audioData->samplesRecordBufferSize);
	this->audioData->paths = new audioPaths();
	this->audioData->paths->ptr = NULL;
	this->audioData->paths->size = 0;
	this->audioData->paths->mutex = new std::mutex();
	this->audioData->volume = 30.0f;

	double data[2] = {0, 0};
	try {
		this->audioApi->openStream(this->streamParams->oParams, this->streamParams->iParams, SAMPLE_FORMAT, this->sample_rate,
			this->streamParams->bufferFrames, &processAudio, (void *)this->audioData, this->streamParams->options);
	}
	catch (RtAudioError& e) {
		e.printMessage();
		exit(0);
	}

	try {
		this->audioApi->startStream();
	}
	catch (RtAudioError& e) {
		e.printMessage();
		exit(0);
	}

	this->currentPaths = new audioPaths();
	this->currentPaths->ptr = NULL;
	this->currentPaths->size = 0;
	this->currentPaths->mutex = new std::mutex;

	//Create Rs vector
	// aca agregamos (audio_length + 1) - SG 13/04
	this->audioData->Rs = new std::vector<float>((audio_length + 1) * this->sample_rate);
}

void AudioRenderer::resetStream() {
	if (this->audioApi->isStreamOpen()) {
		try {
			this->audioApi->closeStream();
		}
		catch (RtAudioError& e) {
			e.printMessage();
			exit(0);
		}
	}

	memset(this->audioData->samplesRecordBuffer->buffer, 0, this->audioData->samplesRecordBufferSize);
	this->audioData->samplesRecordBuffer->head = 0;
	this->audioData->samplesRecordBuffer->tail = 0;
	this->audioData->samplesRecordBuffer->insertSampleElements((SAMPLE_TYPE*)&this->audio_sample_file.samples[0][0], this->sample_rate, this->audio_sample_file.samples[0].size());
	this->audioData->samplesRecordBuffer->tail = this->sample_rate - 1;



	try {
		this->audioApi->openStream(this->streamParams->oParams, NULL, SAMPLE_FORMAT, this->sample_rate,
			this->streamParams->bufferFrames, &processAudioSample, (void *)this->audioData, this->streamParams->options);
	}
	catch (RtAudioError& e) {
		e.printMessage();
		exit(0);
	}
	try {
		this->audioApi->startStream();
	}
	catch (RtAudioError& e) {
		e.printMessage();
		exit(0);
	}

}

void AudioRenderer::render(Scene * scene, Camera * camera, Source * source) {
	RayTracer rt = RayTracer(scene, camera->pos, this->listener_size, source->pos, this->source_power, this->currentPaths, this->max_reflexions, 1-(this->absorbtion_coef), this->num_rays);
	rt.OmnidirectionalUniformSphereRayCast();
	
	//Initialize Rs
	std::fill(this->audioData->Rs->begin(), this->audioData->Rs->end(), 0.0);

	//Paths store the distance, to get the corresponding cell in vector Rs we need to find the elapsed time
	for (int i = 0; i < this->currentPaths->size; i++) {
		float distance = this->currentPaths->ptr[i].travelled_distance;
		float remaining_factor = this->currentPaths->ptr[i].remaining_energy_factor;
		float elapsed_time = distance / SPEED_OF_SOUND;
		//The elapsed time is then converted to a position in the array by multiplying the time by the samples per second
		//This way a path that takes 1s to reach the listener will ocuppy the last position in the array.
		unsigned int array_pos = round(elapsed_time * this->sample_rate);
		if (array_pos < this->audioData->samplesRecordBufferSize && array_pos >= 0) {
			(*this->audioData->Rs)[array_pos] += remaining_factor;
		}
	}

	
	// codigo que estaba comentado y fallaba - SG 13/04
	// std::ofstream rs_file("rs.txt");
	// rs_file << std::setprecision(7);
	// float received_energy = 0;
	// for (int i = 0; i < this->audioData->samplesRecordBufferSize; i++) {
	// 	rs_file << (*this->audioData->Rs)[i] << ",";
	// 	received_energy += (*this->audioData->Rs)[i];
	// }
	// rs_file << std::endl << received_energy;
	// rs_file.close();
}

void AudioRenderer::updateVolume(float value) {
	this->audioData->volume += value;
	std::cout << this->audioData->volume << std::endl;
}

AudioRenderer::~AudioRenderer() {

}