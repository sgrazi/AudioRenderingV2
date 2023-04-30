// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pch.h"
#include <iostream>
#include <io.h>

#include <ctime>
#include <windows.h>
#include "Utils.h"
#include "Camera.h"
#include "Mesh.h"
#include "ShaderProgram.h"

#include <embree3/rtcore.h>
#include <embree3/rtcore_common.h>
#include <stdio.h>
#include <math.h>
#include <limits>

#include "Scene.h"
#include "AudioRenderer.h"
#include "AudioFileRenderer.h"
#include "tinyxml2.h"

#if defined(_WIN32)
#  include <conio.h>
#  include <windows.h>
#endif

using namespace std;

#include "AppModes.h"

void init();
void initGL();
void draw();
void close();

SDL_Window* window = NULL;
SDL_GLContext context;

int WIDTH = 800;
int HEIGHT = 600;

/*
 * A minimal tutorial.
 *
 * It demonstrates how to intersect a ray with a single triangle. It is
 * meant to get you started as quickly as possible, and does not output
 * an image.
 *
 * For more complex examples, see the other tutorials.
 *
 * Compile this file using
 *
 *   gcc -std=c99 \
 *       -I<PATH>/<TO>/<EMBREE>/include \
 *       -o minimal \
 *       minimal.c \
 *       -L<PATH>/<TO>/<EMBREE>/lib \
 *       -lembree3
 *
 * You should be able to compile this using a C or C++ compiler.
 */

 /*
  * This is only required to make the tutorial compile even when
  * a custom namespace is set.
  */
#if defined(RTC_NAMESPACE_USE)
RTC_NAMESPACE_USE
#endif

// Two-channel sawtooth wave generator.
int saw(void* outputBuffer, void* inputBuffer, unsigned int nBufferFrames,
	double streamTime, RtAudioStreamStatus status, void* userData)
{
	unsigned int i, j;
	double* buffer = (double*)outputBuffer;
	double* lastValues = (double*)userData;
	if (status)
		std::cout << "Stream underflow detected!" << std::endl;
	// Write interleaved audio data.
	AudioFile<float> audio;
	const char* file_path = "assets/sound_samples/experimento_entrada_16KHz.wav";
	//if (_access(file_path, 0) == 0) {
	//	// file exists
	//	printf("existe la mierda esta\n");
	//}
	audio.load("assets/sound_samples/experimento_entrada_16KHz.wav");
	
	for (i = 0; i < nBufferFrames; i++) {
		*buffer++ = (double) audio.samples.at(0).at(i);
	}
	return 0;
}

/*
 * We will register this error handler with the device in initializeDevice(),
 * so that we are automatically informed on errors.
 * This is extremely helpful for finding bugs in your code, prevents you
 * from having to add explicit error checking to each Embree API call.
 */
	void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
	printf("error %d: %s\n", error, str);
}

/*
 * Embree has a notion of devices, which are entities that can run
 * raytracing kernels.
 * We initialize our device here, and then register the error handler so that
 * we don't miss any errors.
 *
 * rtcNewDevice() takes a configuration string as an argument. See the API docs
 * for more information.
 *
 * Note that RTCDevice is reference-counted.
 */
RTCDevice initializeDevice()
{
	RTCDevice device = rtcNewDevice(NULL);

	if (!device)
		printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));

	rtcSetDeviceErrorFunction(device, errorFunction, NULL);
	return device;
}

/*
 * Create a scene, which is a collection of geometry objects. Scenes are
 * what the intersect / occluded functions work on. You can think of a
 * scene as an acceleration structure, e.g. a bounding-volume hierarchy.
 *
 * Scenes, like devices, are reference-counted.
 */
//Scene * initializeScene(RTCDevice device)
//{
//	Scene * new_scene = new Scene(device);
//	new_scene->addObjectFromOBJ("models/street.obj", glm::vec3(0.0f,0.0f,0.0f), 10.0f, &device);
//	//new_scene->addMeshFromObj("models/teapot.obj", device);
//	new_scene->commitScene();
//	return new_scene;
//}

void waitForKeyPressedUnderWindows()
{
#if defined(_WIN32)
	HANDLE hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);

	CONSOLE_SCREEN_BUFFER_INFO csbi;
	if (!GetConsoleScreenBufferInfo(hStdOutput, &csbi)) {
		printf("GetConsoleScreenBufferInfo failed: %d\n", GetLastError());
		return;
	}

	/* do not pause when running on a shell */
	if (csbi.dwCursorPosition.X != 0 || csbi.dwCursorPosition.Y != 0)
		return;

	/* only pause if running in separate console window. */
	printf("\n\tPress any key to exit...\n");
	int ch = _getch();
#endif
}


/* -------------------------------------------------------------------------- */

void init() {
	SDL_Init(SDL_INIT_VIDEO);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	window = SDL_CreateWindow("OpenGL", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT,
		SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
	context = SDL_GL_CreateContext(window);
	glewExperimental = GL_TRUE;
	glewInit();
	initGL();
}

void initGL() {
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//OpenGL attribs
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
	glDisable(GL_CULL_FACE);
}

void draw() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void close() {
	SDL_DestroyWindow(window);
	window = NULL;
	SDL_Quit();
}

void getFileImpulseResponse(char* file_path, int mode) {
	if (mode == AURALIZE) {
		init();
	}
	RTCDevice device = initializeDevice();
	Scene * scene = new Scene(device, mode);

	tinyxml2::XMLDocument scene_doc;

	if (scene_doc.LoadFile(file_path)) {
		cout << "Error loading file" << endl;
		return;
	}

	const char* model_file_path = scene_doc.FirstChildElement("SCENE")->FirstChildElement("MODEL")->GetText();
	float scene_size = scene_doc.FirstChildElement("SCENE")->FirstChildElement("SIZE")->FloatText();
	int max_reflexions = scene_doc.FirstChildElement("SCENE")->FirstChildElement("MAX_REFLEXIONS")->IntText();
	float absorbtion_coef = scene_doc.FirstChildElement("SCENE")->FirstChildElement("ABSORBTION")->FloatText();
	int num_rays = scene_doc.FirstChildElement("SCENE")->FirstChildElement("NUM_RAYS")->IntText();

	float source_power = scene_doc.FirstChildElement("SCENE")->FirstChildElement("SOURCE")->FirstChildElement("POWER")->FloatText();
	glm::vec3 source_pos = glm::vec3(
		scene_doc.FirstChildElement("SCENE")->FirstChildElement("SOURCE")->FirstChildElement("POS_X")->FloatText(),
		scene_doc.FirstChildElement("SCENE")->FirstChildElement("SOURCE")->FirstChildElement("POS_Y")->FloatText(),
		scene_doc.FirstChildElement("SCENE")->FirstChildElement("SOURCE")->FirstChildElement("POS_Z")->FloatText()
	);

	float listener_size = scene_doc.FirstChildElement("SCENE")->FirstChildElement("LISTENER")->FirstChildElement("SIZE")->FloatText();
	glm::vec3 listener_pos = glm::vec3(
		scene_doc.FirstChildElement("SCENE")->FirstChildElement("LISTENER")->FirstChildElement("POS_X")->FloatText(),
		scene_doc.FirstChildElement("SCENE")->FirstChildElement("LISTENER")->FirstChildElement("POS_Y")->FloatText(),
		scene_doc.FirstChildElement("SCENE")->FirstChildElement("LISTENER")->FirstChildElement("POS_Z")->FloatText()
	);

	int sample_rate;
    if (mode == AURALIZE){
        if (scene_doc.FirstChildElement("SCENE")->FirstChildElement("OUT_SAMPLERATE")) {
            sample_rate = scene_doc.FirstChildElement("SCENE")->FirstChildElement("OUT_SAMPLERATE")->IntText();
        }
        else {
            sample_rate = SAMPLE_RATE;
        }
    }

	scene->addObjectFromOBJ(model_file_path, glm::vec3(0.0f, 0.0f, 0.0f), scene_size, &device);
	scene->commitScene();
	
    if (mode == SIMULATE){
        const char* measurement_file_path = scene_doc.FirstChildElement("SCENE")->FirstChildElement("MEASUREMENT")->FirstChildElement("FILE")->GetText();
        unsigned int measurement_length = scene_doc.FirstChildElement("SCENE")->FirstChildElement("MEASUREMENT")->FirstChildElement("LENGTH")->UnsignedText();

        timeInterval interval;
        
        if (scene_doc.FirstChildElement("SCENE")->FirstChildElement("ANALYZE")) {
            unsigned int begin = scene_doc.FirstChildElement("SCENE")->FirstChildElement("ANALYZE")->FirstChildElement("BEGIN")->IntText();
            unsigned int end = scene_doc.FirstChildElement("SCENE")->FirstChildElement("ANALYZE")->FirstChildElement("END")->IntText();
            interval = { begin, end };
        }
        else {
            interval = { 0, 0 };
        }

        renderAudioFile(scene, listener_pos, listener_size, source_pos, source_power, measurement_file_path, measurement_length, max_reflexions, absorbtion_coef, num_rays, interval);
    } else if (mode == AURALIZE) {
        const char * sound_sample = NULL;
		AudioRenderer audio;
		if (scene_doc.FirstChildElement("SCENE")->FirstChildElement("SOUND_SAMPLE")) {
			sound_sample = scene_doc.FirstChildElement("SCENE")->FirstChildElement("SOUND_SAMPLE")->GetText();
			audio = AudioRenderer(max_reflexions, absorbtion_coef, num_rays, source_power, listener_size, sample_rate, sound_sample);
		}
		else {
			audio = AudioRenderer(max_reflexions, absorbtion_coef, num_rays, source_power, listener_size, sample_rate);
		}
		Camera cam = Camera(listener_pos, WIDTH, HEIGHT, 45, window);
		Source * source = new Source(glm::vec3(0.0f, 0.0f, 0.0f), 0.25, "assets/models/sphere.obj");
		//audio.render(scene, &cam, source);
		ShaderProgram* pass = new ShaderProgram("assets/shaders/pass.vert", "assets/shaders/pass.frag");
		bool exit = false;

		bool wireframe = false;
		bool active_rendering = false;

		SDL_Event event;

		double frameTime = 1000.0f / 65.0f;

		std::clock_t start;
		while (!exit) {
			start = clock();
			while (SDL_PollEvent(&event) != 0) {
				switch (event.type) {
				case SDL_QUIT:
					exit = true;
					break;
				case SDL_KEYDOWN: {
					if (event.key.keysym.sym == SDLK_ESCAPE) {
						exit = true;
						break;
					}
					else if (event.key.keysym.sym == SDLK_m) {
						cam.moveCam();
						break;
					}
					else if (event.key.keysym.sym == SDLK_e) {
						source->pos = cam.pos;
						break;
					}
					else if (event.key.keysym.sym == SDLK_r) {
						audio.render(scene, &cam, source);
						break;
					}
					else if (event.key.keysym.sym == SDLK_t) {
						active_rendering = true;
						break;
					}
					else if (event.key.keysym.sym == SDLK_j) {
						audio.updateVolume(-1.0);
						break;
					}
					else if (event.key.keysym.sym == SDLK_k) {
						audio.updateVolume(1.0);
						break;
					}
					else if (event.key.keysym.sym == SDLK_SPACE) {
						audio.resetStream();
						break;
					}

				}
								break;
				case SDL_MOUSEBUTTONDOWN: {
				}
										break;
				}
			}

			//cam.update();
			if (active_rendering) {
				audio.render(scene, &cam, source);
			}
			//draw();
			//pass->bind();
			//GLuint colorID = glGetUniformLocation(pass->getId(), "in_color");
			//glUniform4fv(colorID, 1, &(glm::vec4(1, 1, 1, 1)[0]));
			////ViewProjectionMatrix
			//GLuint worldTransformID = glGetUniformLocation(pass->getId(), "worldTransform");
			//glUniformMatrix4fv(worldTransformID, 1, GL_FALSE, &cam.modelViewProjectionMatrix[0][0]);
			//GLuint modelMatrixID = glGetUniformLocation(pass->getId(), "modelMatrix");
			//GLuint vistaID = glGetUniformLocation(pass->getId(), "vista");
			//glUniform3fv(vistaID, 1, &((cam.ref - cam.pos)[0]));
			////Directional light. To do a point light more shader code is needed.
			//GLuint lightDirID = glGetUniformLocation(pass->getId(), "lightDir");
			//glUniform3fv(lightDirID, 1, &(glm::vec3(1, -1, 1)[0]));
			//for (int i = 0; i < scene->objects.size(); ++i) {
			//	glUniformMatrix4fv(modelMatrixID, 1, GL_FALSE, &scene->objects[i]->getModelMatrix()[0][0]);
			//	scene->objects[i]->draw();
			//}
			//glUniformMatrix4fv(modelMatrixID, 1, GL_FALSE, &source->getModelMatrix()[0][0]);
			//glUniform4fv(colorID, 1, &(glm::vec4(1,0,0,1)[0]));
			//source->draw();
			//pass->unbind();

			//double dif = frameTime - ((clock() - start) * (1000.0 / double(CLOCKS_PER_SEC)));
			//if (dif > 0) {
			//	Sleep(int(dif));
			//}
			//SDL_GL_SwapWindow(window);
		}

		delete(scene);
		/* Though not strictly necessary in this example, you should
		/* always make sure to release resources allocated through Embree. */
		rtcReleaseDevice(device);
		/* wait for user input under Windows when opened in separate window */
		waitForKeyPressedUnderWindows();

		close();
    }
}

int main(int argc, char* argv[]) {
	RtAudio dac;
	if (dac.getDeviceCount() < 1) {
		std::cout << "\nNo audio devices found!\n";
		exit(0);
	}
	RtAudio::StreamParameters parameters;
	parameters.deviceId = dac.getDefaultOutputDevice();
	parameters.nChannels = 1; // tienq ue machear con los channels del audio
	parameters.firstChannel = 0;
	

	AudioFile<float> audio;
	const char* file_path = "assets/sound_samples/experimento_entrada_16KHz.wav";
	audio.load(file_path);
	unsigned int sampleRate = audio.getSampleRate() / audio.samples.size();
	unsigned int bufferFrames = audio.samples.at(0).size(); // 256 sample frames
	const int length = audio.samples.at(0).size();
	double data[1];
	try {
		dac.openStream(&parameters, NULL, RTAUDIO_FLOAT64,
			sampleRate, &bufferFrames, &saw, (void*)&data);
		dac.startStream();
	}
	catch (RtAudioError& e) {
		e.printMessage();
		exit(0);
	}

	char input;
	std::cout << "\nPlaying ... press <enter> to quit.\n";
	std::cin.get(input);
	try {
		// Stop the stream
		dac.stopStream();
	}
	catch (RtAudioError& e) {
		e.printMessage();
	}
	if (dac.isStreamOpen()) dac.closeStream();
	return 0;
	/*char* mode_str = argv[1];
    int mode;
	if (!strcmp(mode_str, "simulate")) {
        mode = SIMULATE;
		cout << "Simulating audio" << endl;
	}
	else if (!strcmp(mode_str, "auralize")) {
        mode = AURALIZE;
		cout << "Auralizing audio" << endl;
	}
	else {
		cout << "Invalid mode" << endl;
	}
    char* file_path = argv[2];
    getFileImpulseResponse(file_path, mode);
	return 0;*/
}