#include "./Utils.h"

glm::vec3 gdt2glm(gdt::vec3f vector) {
	return glm::vec3(vector[0], vector[1], vector[2]);
}

double median(std::vector<double> values)
{
	size_t size = values.size();

	if (size == 0)
	{
		return 0;
	}
	else
	{
		sort(values.begin(), values.end());
		if (size % 2 == 0)
		{
			return (values[size / 2 - 1] + values[size / 2]) / 2;
		}
		else
		{
			return values[size / 2];
		}
	}
}

float distanceP2P(gdt::vec3f p1, gdt::vec3f p2)
{
	return std::sqrt(std::pow((p2.x - p1.x), 2) + std::pow((p2.y - p1.y), 2) + std::pow((p2.z - p1.z), 2));
}