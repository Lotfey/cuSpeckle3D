#ifndef    PARAMALGO_H
#define    PARAMALGO_H


template <typename T>
struct vec3D 
{
    T x;
    T y;
    T z;
};

template <typename T>
struct paramAlgo {
    int N0;
    T alpha;
    typedef T dataType;      // this means that dataType and T are the same type
    int NMCmax;
};

template <typename T>
struct paramSensor
 {
    int nbit;
    T sigma;
    vec3D<int> dims;
};

template <typename T>
struct paramSpeckle
 {
    char distribR;
    T mu;
    T lambda;
    T gamma;
    T sigmaR=0;
};


///////// structures for MC estimation /////////
struct Random_sphere 
{
    float x;
    float y;
    float z;
    float r;
};

struct Boolean_model_sphere
 {
    Random_sphere sphere;
    float rbound;
};



#endif