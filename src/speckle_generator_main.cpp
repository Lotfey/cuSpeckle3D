#include "io_png.h"
#include "util.h"
#include "boolean_model.h"

//#define DEBUG

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        show_help();
        return -1;
    }

    //show help
    if (cmdOptionExists(argv, argv + argc, "-h"))
    {
        show_help();
        return (-1);
    }

    //get file name
    std::string fileNameOut(argv[1]);

    float mu, sigma, sigmaR, gamma, alpha, lambda, delta;
    int width, height, depth, nbit, N0, NMCmax;
    unsigned int seed;
    char distribR;


    /**************************************************/
    /*************   GET INPUT OPTIONS   **************/
    /**************************************************/

    //algorithm params
    //
    if (cmdOptionExists(argv, argv + argc, "-alpha"))
    {
        alpha = (float)atof(getCmdOption(argv, argv + argc, "-alpha"));
    }
    else
        alpha = 0.1;

    if (cmdOptionExists(argv, argv + argc, "-N0"))
    {
        N0 = (int)atoi(getCmdOption(argv, argv + argc, "-N0"));
    }
    else
        N0 = 1001;

    if (cmdOptionExists(argv, argv + argc, "-NMCmax"))
    {
        NMCmax = (int)atoi(getCmdOption(argv, argv + argc, "-NMCmax"));
    }
    else
        NMCmax = (int)pow(2, 24);

    //sensor params
    if (cmdOptionExists(argv, argv + argc, "-width"))
    {
        width = (int)atoi(getCmdOption(argv, argv + argc, "-width"));
    }
    else
        width = 500;

    if (cmdOptionExists(argv, argv + argc, "-height"))
        height = (int)atoi(getCmdOption(argv, argv + argc, "-height"));
    else
        height = 500;
    
    if (cmdOptionExists(argv, argv + argc, "-depth"))
        depth = (int)atoi(getCmdOption(argv, argv + argc, "-depth"));
    else
        depth = 500;
    

    if (cmdOptionExists(argv, argv + argc, "-nbit"))
        nbit = (int)atoi(getCmdOption(argv, argv + argc, "-nbit"));
    else
        nbit = 8;

    if (cmdOptionExists(argv, argv + argc, "-sigmaG"))
    {
        sigma = (float)atof(getCmdOption(argv, argv + argc, "-sigmaG"));
    }
    else
        sigma = 1;

    if (cmdOptionExists(argv, argv + argc, "-seed"))
    {
        int seed_choice = atoi(getCmdOption(argv, argv + argc, "-seed"));
        if (seed_choice == 0)
        {
            // generate a random seed
            std::random_device rd;
            seed = (unsigned int)rd();
        }
        else
        {
            seed = (unsigned int)seed_choice;
        }
    }
    else
        seed = 2020u;

    //speckle params
    if (cmdOptionExists(argv, argv + argc, "-distribR"))
    {
        distribR = getCmdOption(argv, argv + argc, "-distribR")[0];
    }
    else
        distribR = 'E';

    if (cmdOptionExists(argv, argv + argc, "-gamma"))
    {
        gamma = (float)atof(getCmdOption(argv, argv + argc, "-gamma"));
    }
    else
        gamma = 0.6;

    if (cmdOptionExists(argv, argv + argc, "-lambda"))
    {
        lambda = (float)atof(getCmdOption(argv, argv + argc, "-lambda"));
    }
    else
        lambda = round(5 / 9.0 * width * height*depth); //500000;
        //lambda = round( 0.3f * width * height* depth);
        //lambda = round( 0.1f * width * height* depth);

    if (cmdOptionExists(argv, argv + argc, "-sigmaR"))
    {
        sigmaR = (float)atof(getCmdOption(argv, argv + argc, "-sigmaR"));
    }
    else
        sigmaR = 0;

    if (cmdOptionExists(argv, argv + argc, "-mu"))
    {
        mu = (float)atof(getCmdOption(argv, argv + argc, "-mu"));
    }
    else
        mu = 0.5;

    // Parameters initialisation 
    paramSpeckle<float> myParamSpeckle;
    paramAlgo<float> myParamAlgo;
    paramSensor<float> myParamSensor;
    vec3D<int> myDims;

    myDims.x = height;
    myDims.y = width;
    myDims.z = depth;

    myParamSpeckle.distribR = distribR;
    myParamSpeckle.gamma = gamma;
    myParamSpeckle.lambda = lambda;
    myParamSpeckle.mu = mu;
    myParamSpeckle.sigmaR = sigmaR;

    myParamAlgo.alpha = alpha;
    myParamAlgo.N0 = N0;
    myParamAlgo.NMCmax = NMCmax;

    myParamSensor.dims = myDims;
    myParamSensor.nbit = nbit;
    myParamSensor.sigma = sigma;

    //display parameters
    std::cout << "------- Sensor params -------" << std::endl;
    std::cout << "Image size [height x width x depth] : " << height << " x " << width << " x " << depth <<  std::endl;    
    std::cout << "nbit : " << nbit << std::endl;
    std::cout << "sigma : " << sigma << std::endl;

    std::cout << "------- Speckle params -------" << std::endl;
    std::cout << "distribR : " << distribR << std::endl;
    std::cout << "gamma : " << gamma << std::endl;
    std::cout << "lambda : " << lambda << std::endl;
    std::cout << "mu : " << mu << std::endl;
    std::cout << "sigmaR : " << sigmaR << std::endl;
    
    std::cout << "------- Algorithm params -------" << std::endl;
    std::cout << "alpha : " << alpha << std::endl;
    std::cout << "N0 : " << N0 << std::endl;
    std::cout << "NMCmax : " << NMCmax << std::endl;
    std::cout << "precision : " << "float" << std::endl;
    std::cout << "------- Seed -------" << std::endl;
    std::cout << "seed : " << seed << std::endl;
    /**************************************************/
    /******** Debugging print out parameters    *******/
    /******** and variable to check the program *******/
    /********           correctness             *******/ 
    /**************************************************/
     
    /**************************************************/
    /*****  TIME AND CARRY OUT GRAIN RENDERING   ******/
    /**************************************************/

    struct timeval start, end;
    gettimeofday(&start, NULL);

    //execute the speckle generation
    std::cout << "***************************" << std::endl;

    float *speckle_matrix;
    speckle_matrix = (float *)malloc(width * height * depth * sizeof(float));

    // Draw the number of spheres, following a Poisson distribution of intensity "lambda"
    std::mt19937_64 generator(seed);
    std::poisson_distribution<int> distrib(myParamSpeckle.lambda);
    int number = distrib(generator);
        
    //int number =2;   
    std:: cout<< "number of spheres: "<< number<< std::endl;
    
    // allocate memory for the generated Random_radius
    float *Random_radius;
    Random_radius = (float *)malloc(number * sizeof(float));
    // Random_radius[0]= 8.0f;
    // Random_radius[1]= 8.0f;
    
    
    // generate RBound
    float *RBound;
    RBound = (float *)malloc(number * sizeof(float)); // same size as Random_radius

    // float tmp0_icdf = icdf(1 - pi * alpha / sqrt(2) / pow(2, nbit) / 10);
    // RBound[0] = pow(Random_radius[0] + (1 + delta) * tmp0_icdf * sigma, 2);
    // RBound[1] = pow(Random_radius[1] + (1 + delta) * tmp0_icdf * sigma, 2);
    
    // generate RC
    float *Random_centers;
    Random_centers = (float *)malloc(3 * number * sizeof(float)); // this is a vector with x, y  and z for each RC
    // Random_centers[0]=10;
    // Random_centers[1]=20;
    // Random_centers[2]=30;
    
    // Random_centers[3]=15;
    // Random_centers[4]=25;
    // Random_centers[5]=30;

    boolean_model(Random_centers, Random_radius, RBound, myParamSpeckle, myParamAlgo, myParamSensor, number, seed);

    monte_carlo_estimation_cuda(speckle_matrix,
                                Random_centers,
                                Random_radius,
                                RBound,
                                number, 
                                seed, 
                                width, height,depth, 
                                alpha, nbit, gamma, N0, sigma);

    //quantization ==> output in range [0, 2^nbit-1]
    float *qt_out = NULL;
    qt_out = (float *)malloc(width * height * depth * sizeof(float));
    quantization(qt_out, speckle_matrix, width * height * depth, nbit);

    //create output float image
    float *imgOut = NULL;
    imgOut = (float *)malloc(width * height *depth * 1 * sizeof(float)); // number of channels is 3 <=> RGB

    for (int i=0; i< height*width*depth; ++i)
    {
        imgOut[i]=qt_out[i]; // red
    }
    
    //speckle_matrix[Random_centers[1] * width +Random_centers[0] + Random_centers[2] * ( width * height)]=255;
    write_csv_matrix(fileNameOut, imgOut, height, width, depth);
    write_output_image(imgOut, fileNameOut, myParamSpeckle, myParamAlgo, myParamSensor);
    
    int chanel=1;
    std::string fileNameOut1=fileNameOut+ "_" + typeid(imgOut[0]).name() + "_" +
						std::to_string(chanel) + "_" +
						std::to_string(height) + "x" +
						std::to_string(width)  + "x" +
	
    					std::to_string(depth)  +".raw";
    // This fumction write the memory bytes
    writeRawFile(fileNameOut1, imgOut, width*height*depth);

    // free the allocated memory
    free(imgOut);
    free(qt_out);
    free(speckle_matrix);
    free(Random_radius);
    free(RBound);
    free(Random_centers);

    // display elapsed time
    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - start.tv_sec) +
                         (end.tv_usec - start.tv_usec) / 1.e6;
    std::cout << "\nTime elapsed : " << elapsedTime/60<< " minutes" << std::endl;

        double vm, rss;
   process_mem_usage(vm, rss);
   std::cout << "Virtual Memory [VM]: " << vm << " KB"<<
             "\nResident set size [RSS]: " << rss << " KB"<< std::endl;

    std::cout << "***************************" << std::endl;
    return (0);
}
