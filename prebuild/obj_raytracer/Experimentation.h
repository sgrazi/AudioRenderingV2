#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>

struct FileData {
	std::string name;
	double maximum_value;
};

class Experimentation
{
private:
	static Experimentation* instance;
	Experimentation();
	std::vector<FileData> files;

public:
	static Experimentation* getInstance();

	static void add_file_data(FileData file_data);

	static void results();
};
