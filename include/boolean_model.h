#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include <cmath>
#include <functional>

#include "util.h"
#include "data_structures.h"
#include <boost/math/special_functions/erf.hpp>
// #include "mapping_functions.h"

const float pi = 3.14159265358979323846f;

float estimate_delta(vec3D<int> dims);

float icdf(float p);
void boolean_model(float *Random_centers, float *Random_radius, float *RBound,
                     paramSpeckle<float> myParamSpeckle,
                     paramAlgo<float> myParamAlgo,
                     paramSensor<float> myParamSensor,
                     int number, unsigned int seed);

int monte_carlo_estimation_cuda(float *speckle_matrix, float *Random_centers, float *Random_radius,
                                float *RBound, int number, unsigned int seed,
                                int width, int height,int depth,
                                float alpha, int nbit, float gamma, int N0, float sigma);

void quantization(float *img_out, float *speckle_matrix, int img_size, int nbit);