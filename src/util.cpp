

#include "io_png.h"
#include "util.h"





/// help on usage of inpainting code
void show_help()
{
    std::cerr << "\nSpeckle Generator.\n"
              << "Usage: "
              << " speckle_generator_main imgNameOut.png [options]\n\n"
              << "Options (default values in parentheses)\n"
              << "-prec : data precision (0: float(default), 1:double)\n"
              << "-distribR : probability distribution of the radii ('E' for exponential, 'U' for uniform, 'P' for Poisson, 'L' for log-normal)\n"
              << "-gamma : standard deviation of the radius\n"
              << "-lambda : average number of disks per image\n"
              << "-sigmaR : standard deviation of radius (for 'l' only)\n"
              << "-mu : average radius of disks\n"
              << "-alpha : quantization error probability\n"
              << "-N0 : sample size to estimate s^2\n"
              << "-NMCmax : size of the largest MC sample \n"
              << "-width : output image width\n"
              << "-height : output image height\n"
              << "-nbit : bit depth\n"
              << "-sigmaG : standard deviation of the Gaussian PSF\n"
              << "-seed : seed of the random generator (default: 2020, shuffle: 0)\n"
              << std::endl;
}

/**
 * 
 */
/**
* @brief Find the command option named option
*/
char *getCmdOption(char **begin, char **end, const std::string &option)
{
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

/**
 * 
 */
/**
* @brief Check for input parameter
*
* @param beginning of input command chain
* @param end of input command chain
* @return whether the parameter exists or not
*/
bool cmdOptionExists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

/**
 * 
 */
/**
* @brief Get file exension of file name
*
* @param File name
* @return File extension
*/
std::string getFileExt(const std::string &s)
{
    size_t i = s.rfind('.', s.length());
    if (i != std::string::npos)
    {
        return (s.substr(i + 1, s.length() - i));
    }
    else
        return ("");
}

/**
 * 
 */
/**
* @brief Get file name, without extension
*
* @param Input file name
* @return File name without extension
*/
std::string getFileName(const std::string &s)
{
    size_t i = s.rfind('.', s.length());
    if (i != std::string::npos)
    {
        return (s.substr(0, i));
    }
    else
        return (s);
}

/**
 * 
 */
/**
* @brief Get current directory
*
* @return Current directory name
*/
std::string get_curr_dir()
{
    size_t maxBufSize = 1024;
    char buf[maxBufSize];
    char *charTemp = getcwd(buf, maxBufSize);
    std::string currDir(charTemp);
    return (currDir);
}

/**
 * 
 */
/**
* @brief Write the output to a .png image.
*
* @param imgOut output image to write
* @param fileNameOut output file name
* @return 0 if write success, -1 if failure
*/
int write_output_image(float *imgOut, const std::string fileNameOut,
                       paramSpeckle<float> myParamSpeckle, paramAlgo<float> myParamAlgo, paramSensor<float> myParamSensor)
{
    std::string outputExtension(getFileExt(fileNameOut));

    //write output image
    std::string fileNameOutFull = (char *)((getFileName(fileNameOut) + "." + outputExtension).c_str());
    std::cout << "output file name : " << fileNameOutFull << std::endl;

    if (strcmp((const char *)(outputExtension.c_str()), "png") == 0 ||
        strcmp((const char *)(outputExtension.c_str()), "") == 0) //png files
    {
        if (0 != io_png_write_f32(fileNameOutFull.c_str(), imgOut,
                                  myParamSensor.dims.x, myParamSensor.dims.y, 1)) // 1 channel is used
        {
            std::cout << "Error, could not write the image file." << std::endl;
            return (-1);
        }
    }
    else
    {
        std::cout << "Error, unknown output file extension." << std::endl;
        return (-1);
    }

    return (0);
}




void write_csv_matrix(std:: string filename, float* data, int height , int width , int depth )
{
	std::ofstream myfile(filename);
    myfile << height  << "\n";
    myfile << width << "\n";
    myfile << depth  << "\n";


	for (int n = 0; n < height* width*depth; n++)
	myfile << std::setprecision(16)<< data[n] << "\n";
	
}

void write_csv_centers(std:: string filename, float* data, int size)
{
    std::ofstream myfile(filename);
    
    // save sphere centers into .csv file of size (size/3 x 3)
    // each row represente a sphere of center x, y, z.  
    for (int i=0; i<size; ++i)
    {
        myfile << std::setprecision(16) << data[3 * i]<<",";
        myfile << std::setprecision(16) << data[3 * i+ 1]<<",";
        myfile << std::setprecision(16) << data[3 * i+ 2]<<"\n";
                
    }
}

void write_csv_radius(std:: string filename, float* data, int size)
{
    std::ofstream myfile(filename);
    
    // save sphere radius into .csv file of size (size x 1)
    // each row represente a sphere radius Rc.  
    for (int i=0; i<size; ++i)
    {
        myfile << std::setprecision(16) << data[i]<<"\n";
                
    }
}