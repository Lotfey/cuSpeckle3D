#include "io_png.h"
#include "util.h"

/// help on usage of inpainting code
void show_help()
{
    std::cerr << "\nSpeckle Generator.\n"
              << "Usage: "
              << " speckle_generator_main volumeFilename.raw [options]\n\n"
              << "Options (default values in parentheses)\n"
              << "-prec : data precision (float(default))\n"
              << "-distribR : probability distribution of the radii ('E' for exponential, 'U' for uniform, 'P' for Poisson, 'L' for log-normal)\n"
              << "-gamma : standard deviation of the radius\n"
              << "-lambda : average number of disks per volume\n"
              << "-sigmaR : standard deviation of radius (for 'l' only)\n"
              << "-mu : average radius of disks\n"
              << "-alpha : quantization error probability\n"
              << "-N0 : sample size to estimate s^2\n"
              << "-NMCmax : size of the largest MC sample \n"
              << "-width : output volume width\n"
              << "-height : output volume height\n"
              << "-depth : output volume depth\n"
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






// write data into with precision of 16 bits
void write_csv_matrix(std:: string filename, float* data, int height , int width , int depth )
{
	std::ofstream myfile(filename);
    myfile << height  << "\n";
    myfile << width   << "\n";
    myfile << depth   << "\n";

	for (int n = 0; n < height* width*depth; n++)
	myfile << std::setprecision(16)<< data[n] << "\n";
	
}

// Save spheres data
void write_csv_centers(std:: string filename, float* data, int size)
{
    filename=filename+".csv";
    std::ofstream myfile(filename);
    
    // save sphere centers into .csv file of size (size/3 x 3)
    // each row represente a sphere of center x, y, z.  
    for (int i=0; i<size; ++i)
    {
        myfile << std::setprecision(16) << data[3 * i]   <<",";
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

//write raw file this fnction has been tested by itself
void writeRawFile(std::string& fileName, float* dataArray, unsigned size)

{

	std::ofstream myfile;
	myfile.open(fileName, std::ios::out | std::ios::binary);

	if (myfile.is_open())
	{
		myfile.write(reinterpret_cast<const char*>(dataArray), size * sizeof(dataArray[0]));

		myfile.close();
	}
	else
	{
		std::cout << "ERROR! fail to open file from writeRawFile function... \n \n";
	}
		
}

void readRawFile(std::string& fileName, float* dataArray, int size)
{
	std::ifstream myfile;
	myfile.open(fileName, std::ios::in | std::ios::binary);
	if (myfile.is_open())
	{
		myfile.read(reinterpret_cast<char*>(dataArray), size * sizeof(dataArray[0]));
		myfile.close();
	}
	else
	{
		std::cout << "ERROR! fail to open file readRawFile function... \n";
	}

}



void process_mem_usage(double& vm_usage, double& resident_set)
{
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0.0;
   resident_set = 0.0;

   // 'file' stat seems to give the most reliable results
   
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   //
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
  
   unsigned long vsize;
   long rss;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   stat_stream.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
   vm_usage     = vsize / 1024.0;
   resident_set = rss * page_size_kb;
}
