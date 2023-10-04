#include <iostream>
#include <fstream>
#include <string>
#include <vector>

void loadFloatsFromFile(float *floatArray, const std::string &filePath)
{
  std::ifstream file(filePath);
  std::string line;
  std::vector<float> numbers;

  if (file.is_open())
  {
    while (std::getline(file, line))
    {
      try
      {
        float number = std::stof(line);
        numbers.push_back(number);
      }
      catch (const std::invalid_argument &e)
      {
        std::cerr << "Error parsing line: " << line << std::endl;
      }
    }
    file.close();

    // Copy the parsed numbers into the provided float array
    for (size_t i = 0; i < numbers.size(); ++i)
    {
      floatArray[i] = numbers[i];
    }
  }
  else
  {
    std::cerr << "Unable to open file: " << filePath << std::endl;
  }
}

int main()
{
  // cargar los samples del audio nuestro + nuestro ir
  const int sample_size = 128000;
  const int ir_size = 40000;

  std::string samplesFilePath = "./build/obj_raytracer/samples.txt";
  std::string irFilePath = "./build/obj_raytracer/output_ir.txt";
  std::string outputFilePath = "./build/obj_raytracer/output_cameelo.txt";

  float *samples = new float[sample_size];
  float *ir = new float[ir_size];
  float *output = new float[sample_size];

  loadFloatsFromFile(samples, samplesFilePath);
  loadFloatsFromFile(ir, irFilePath);

  // pasamos esos arreglos al algoritmo de camilo

  for (int i = 0; i < ir_size; i++)
  {
    float output_value = 0;
    for (int j = 0; j < ir_size - i; j++)
    {
      output_value += ir[j] * samples[ir_size - 1 - i - j];
    }
    int RvIndex = (ir_size * 2) - 1 - (i * 2);
    ((float *)output)[RvIndex] = output_value;
    ((float *)output)[RvIndex - 1] = output_value;
  }

  // algoritmo original que copie para el de arriba
  /*
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
  */

  std::ofstream outFile(outputFilePath);

  // Check if the file is opened successfully
  if (!outFile.is_open())
  {
    std::cerr << "Error opening the file." << std::endl;
  }
  else
  {
    std::cout << "wrote ir to file" << std::endl;

    // Write each element of the float array to the file, one per line
    for (int i = 0; i < sample_size; ++i)
    {
      outFile << output[i] << std::endl;
    }

    // Close the file
    outFile.close();
  }

  // Return 0 to indicate successful execution
  return 0;
  // testear diff con
  // diff output.txt output_cameelo.txt > diff.txt
}
