#include "Scene.h"
#include "OBJLoader.h"
#include "AppModes.h"
#include <functional>

Scene::Scene(RTCDevice device, int app_mode) {
	this->rtc_scene = rtcNewScene(device);
	this->app_mode = app_mode;
}

static void createEmbreeGeometry(RTCDevice * device, OBJProperites props, RTCScene rtc_scene) {
	RTCGeometry geom = rtcNewGeometry(*device, RTC_GEOMETRY_TYPE_TRIANGLE);

	float* vertices = (float*)rtcSetNewGeometryBuffer(geom,
		RTC_BUFFER_TYPE_VERTEX,
		0,
		RTC_FORMAT_FLOAT3,
		3 * sizeof(float),
		props.vertices.size() / 3); //VERTEX COUNT (3 floats represent 1 vertex)

	std::copy(props.vertices.begin(), props.vertices.end(), vertices);

	unsigned int* indices = (unsigned int*)rtcSetNewGeometryBuffer(geom,
		RTC_BUFFER_TYPE_INDEX,
		0,
		RTC_FORMAT_UINT3,
		3 * sizeof(unsigned int),
		props.indices.size() / 3); //FACE COUNT (3 indices are counted as 1 item since they represent a single triangle)

	std::copy(props.indices.begin(), props.indices.end(), indices);


	rtcCommitGeometry(geom);

	rtcAttachGeometry(rtc_scene, geom);
	rtcReleaseGeometry(geom);
}

void Scene::addObjectFromOBJ(std::string file_name, glm::vec3 pos, float size, RTCDevice * device) {
	//Create mesh in scene
	OBJProperites props = loadOBJ(file_name);

	if (app_mode == AURALIZE){
		SceneObject * object = new SceneObject(pos, size, props);
		this->objects.push_back(object);
	}

	for (int i = 0; i < props.vertices.size(); i = i + 3) {
		props.vertices[i] *= size;
		props.vertices[i + 1] *= size;
		props.vertices[i + 2] *= size;
		props.vertices[i] += pos.x;
		props.vertices[i + 1] += pos.y;
		props.vertices[i + 2] += pos.z;
	}

	if (device) {
		createEmbreeGeometry(device, props, this->rtc_scene);
	}
}


void Scene::commitScene() {
	rtcCommitScene(this->rtc_scene);
}

RTCScene Scene::getRTCScene() {
	return this->rtc_scene;
}

Scene::~Scene() {
	rtcReleaseScene(this->rtc_scene);
	for (int i = 0; i < this->objects.size(); ++i) {
		delete(this->objects[i]);
	}
}