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

void process_file(const std::string& filePath) {
	// Open the file
	std::ifstream file(filePath);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << filePath << std::endl;
		return;
	}

	std::string fileName = filePath;
	size_t pos = filePath.find_last_of("/\\");
	if (pos != std::string::npos) {
		fileName = filePath.substr(pos + 1);
	}


	FileData fileData;
	fileData.name = fileName;

	std::string line;
	double max = 0;
	while (std::getline(file, line)) {
		double value = std::atof(line.c_str());
		if (max < value) max = value;
	}

	fileData.maximum_value = max;

	Experimentation::add_file_data(fileData);

	file.close();
}

void process_files_with_prefix(const std::string& directoryPath, const std::string& prefix) {
	std::string searchPath = directoryPath + "\\" + prefix + "*";
	WIN32_FIND_DATA findFileData;
	HANDLE hFind = FindFirstFile(searchPath.c_str(), &findFileData);

	if (hFind == INVALID_HANDLE_VALUE) {
		std::cerr << "Could not open directory: " << directoryPath << std::endl;
		return;
	}

	do {
		std::string fileName = findFileData.cFileName;
		if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			std::string filePath = directoryPath + "\\" + fileName;
			process_file(filePath);
		}
	} while (FindNextFile(hFind, &findFileData) != 0);

	FindClose(hFind);
}