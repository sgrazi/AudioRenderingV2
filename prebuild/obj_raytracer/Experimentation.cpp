#include "Experimentation.h"

Experimentation* Experimentation::instance = nullptr;

Experimentation::Experimentation() {}

Experimentation* Experimentation::getInstance()
{
	if (!instance)
	{
		instance = new Experimentation();
	}
	return instance;
}

void Experimentation::add_file_data(FileData file_data) {
	instance->files.push_back(file_data);
}

void Experimentation::results() {

	printf("Results from experimentation:\n");
	std::vector<double> max_values;
	for (auto file_data : instance->files) {
		max_values.push_back(file_data.maximum_value);
	}

    if (!max_values.empty()) {
		double sum = std::accumulate(max_values.begin(), max_values.end(), 0.0);
		double mean = sum / max_values.size();

		// Calculate standard deviation
		double sq_sum = std::inner_product(max_values.begin(), max_values.end(), max_values.begin(), 0.0,
			std::plus<double>(), [mean](double x, double y) { return (x - mean) * (y - mean); });
		double standard_deviation = std::sqrt(sq_sum / max_values.size());

		printf("\tMean: %f\n", mean);
		printf("\tStandard deviation: %f\n", standard_deviation);
		printf("\tCoefficient of variation: %f\n", standard_deviation / mean);
    }
}