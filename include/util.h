#include <iostream>
#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include <cstring>
#include <iomanip>     /*setprecision */
#include <fstream>

#include <ios>




#include "data_structures.h"

void show_help();

char *getCmdOption(char **begin, char **end, const std::string &option);

bool cmdOptionExists(char **begin, char **end, const std::string &option);

std::string getFileExt(const std::string &s);

std::string getFileName(const std::string &s);

std::string get_curr_dir();

int write_output_image(float *imgOut, const std::string fileNameOut,
                       paramSpeckle<float> myParamSpeckle,
                       paramAlgo<float> myParamAlgo,
                       paramSensor<float> myParamSensor);
void write_csv_matrix(std:: string filename, float* data,int height , int width , int depth );
void write_csv_centers(std:: string filename, float* data, int size);
void write_csv_radius(std:: string filename, float* data, int size);
void writeRawFile(std::string& fileName, float* dataArray, unsigned size);
void readRawFile(std::string& fileName, float* dataArray, int size);
void process_mem_usage(double& vm_usage, double& resident_set);
