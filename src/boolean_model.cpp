#include "boolean_model.h"



void generate_random_radius(float output[], float *kappa, float *theta,
                            paramSpeckle<float> myParamSpeckle,
                            vec3D<int> dims, 
                            int number, unsigned int seed)
{

    std::mt19937_64 generator(seed);
    float mu = myParamSpeckle.mu;
    float sigmaR = myParamSpeckle.sigmaR;
    float lambda = myParamSpeckle.lambda;
    switch (myParamSpeckle.distribR)
    {
    case 'E':
    {
        printf("Exponential distribution for radii\n");
        std::exponential_distribution<float> distribution(1 / mu); // lambda of the exponential distribution = 1/mu (mu is the mean)
        for (int i = 0; i < number; ++i)
        {
            output[i] = (float)distribution(generator);
        }
        (*kappa) = (float)(1 - exp(-2 * pi * lambda / (float)(dims.x * dims.y) * mu * mu));
        (*theta) = (float)(2 * pi * lambda * mu * exp(-2 * pi * lambda / (float)(dims.x * dims.y) * mu * mu));
        break;
    }
    case 'U':
    {
        printf("Uniform distribution for radii\n");
        std::uniform_real_distribution<float> distribution;
        for (int i = 0; i < number; ++i)
        {
            output[i] = (float)2 * mu * distribution(generator);
        }
        (*kappa) = (float)(1 - exp(-4 * pi * lambda / (float)(dims.x * dims.y) * mu * mu / 3));
        (*theta) = (float)(2 * pi * lambda * mu * exp(-4 * pi * lambda / (float)(dims.x * dims.y) * mu * mu / 3));
        break;
    }
    case 'P':
    {
        printf("Poisson distribution for radii\n");
        std::poisson_distribution<int> distribution(mu); // by default, th type is "int"
        for (int i = 0; i < number; ++i)
        {
            output[i] = (float)distribution(generator);
        }
        (*kappa) = (float)(1 - exp(-2 * pi * lambda / (float)(dims.x * dims.y) * (mu + mu * mu)));
        (*theta) = (float)(2 * pi * lambda * mu * exp(-2 * pi * lambda / (float)(dims.x * dims.y) * (mu + mu * mu)));
        break;
    }
    case 'L':
    {
        printf("Log-normal distribution for radii\n");
        float meanlogR = log(mu * mu / sqrt(sigmaR * sigmaR + mu * mu));
        float stdlogR = log(mu * mu / sqrt(sigmaR * sigmaR + mu * mu));
        std::lognormal_distribution<float> distribution(meanlogR, stdlogR);
        for (int i = 0; i < number; ++i)
        {
            output[i] = (float)distribution(generator);
        }
        (*kappa) = (float)(1 - exp(-pi * lambda / (float)(dims.x * dims.y) * (sigmaR * sigmaR + mu * mu)));
        (*theta) = (float)(2 * pi * lambda * mu * exp(-pi * lambda / (float)(dims.x * dims.y) * (sigmaR * sigmaR + mu * mu)));
        break;
    }
    }
}



/*
Normal inverse cumulative distribution function (see Algorithm in https://fr.mathworks.com/help/stats/norminv.html#mw_06d38ca9-8aa3-44b8-bb8e-549e1c9b4fde)
Using Inverse ERandom_radiusor Function (erfc_inv) from the Boost library ( for details check https://www.boost.org/doc/libs/1_68_0/libs/math/doc/html/math_toolkit/sf_erf/eRandom_radiusor_inv.html)

+ TO install Boost lib on linux ==> sudo apt-get install libboost-all-dev 
*/

float icdf(float p)
{
    return (float)(-sqrt(2) * boost::math::erfc_inv(2 * p));
}


/*

@Random_radius: output
*/

void boolean_model(float *Random_centers,
                    float *Random_radius,
                    float *RBound,
                    paramSpeckle<float> myParamSpeckle,
                    paramAlgo<float> myParamAlgo, 
                    paramSensor<float> myParamSensor, 
                    int number, unsigned int seed)
{
    // init local variables
    vec3D<int> dims = myParamSensor.dims;
    float alpha     = myParamAlgo.alpha;
    int nbit        = myParamSensor.nbit;
    float gamma     = myParamSpeckle.gamma;
    float sigma     = myParamSensor.sigma;

    // translate "Random_centers = repmat(dims,[number,1]).*rand(number,2,prec);"
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> distribution;
    for (int i = 0; i < number; ++i)
    {
        Random_centers[3 * i] =     (float)dims.x * (float)distribution(generator); // x of RC
        Random_centers[3 * i + 1] = (float)dims.y * (float)distribution(generator); // y of RC
        Random_centers[3 * i + 2] = (float)dims.z * (float)distribution(generator); // z of RC
    }

    // write shere centers into .csv file 
    //write_csv_centers("Random_centers.csv", Random_centers, number);

    float kappa = 0, theta = 0;
    // generate random radii
    generate_random_radius(Random_radius, &kappa, &theta, myParamSpeckle, dims, number, seed);

    // write random radius into .csv file 
    //write_csv_radius("Random_radius.csv", Random_radius, number);

    printf("coverage ratio: %f\nexpected perimeter: %f\n", kappa, theta);
    printf("largest Monte Carlo sample size: %d\n",
         (int)floor((float)((float)1 / pi / 2 * gamma * gamma * pow(2, (2 * nbit)) / (alpha * alpha))));

    // estimate delta

    //float delta =0; // delta equal zero for reference image// no displacement involved 
    float delta = estimate_delta(dims);
    printf("Estimated delta from Boolean model: %E\n", delta);

    // estimate B
    float tmp_icdf = icdf(1 - pi * alpha / sqrt(2) / pow(2, nbit) / 10);
    printf("Estimated B : %.4f * sigma\n", tmp_icdf); // This should print 3.7547 for alpha=0.1 and nbit=8 (the same result as Matlab function)

    for (int i = 0; i < number; ++i)
    {
        RBound[i] = pow(Random_radius[i] + (1 + delta) * tmp_icdf * sigma, 2);
    }
/*
#ifdef DEBUG
    printf(" Random_radius samples :");
    printf("\n");
    for (int i = 0; i < 10; ++i)
    {
        std:: cout << Random_radius[i]<< std:: endl;
    }

    printf("Random_centers samples : ");
    printf("\n");
    for (int i = 0; i < 10; ++i)
    {
        std:: cout << Random_centers[i]<< std:: endl;
    }
  
    printf("RBound samples : ");
    printf("\n");
    for (int i = 0; i < 10; ++i)
    {
        std:: cout << Random_centers[i]<< std:: endl;
    }

#endif
*/

}


void quantization(float *img_out, float *speckle_matrix, int img_size, int nbit)
{
    for (int i = 0; i < img_size; ++i)
    {
        img_out[i] = round(speckle_matrix[i]);

        if (img_out[i] < 0)
        {
            img_out[i] = 0;
        }

        if (img_out[i] > pow(2, nbit) - 1)
        {
            img_out[i] = pow(2, nbit) - 1;
        }
    }
}