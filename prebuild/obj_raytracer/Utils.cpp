#include "./Utils.h"

glm::vec3 gdt2glm(gdt::vec3f vector)
{
	return glm::vec3(vector[0], vector[1], vector[2]);
}