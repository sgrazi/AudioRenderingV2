#pragma once

#include <embree3/rtcore.h>
#include <embree3/rtcore_common.h>
#include "Mesh.h"
#include "SceneObject.h"

class Scene {
public:
	RTCScene rtc_scene;
	std::vector<SceneObject*> objects;
	int app_mode;

public:
	Scene() {};
	Scene(RTCDevice device, int app_mode);
	virtual void addObjectFromOBJ(std::string file_name, glm::vec3 pos, float size, RTCDevice * device);
	void commitScene();
	RTCScene getRTCScene();
	~Scene();
};