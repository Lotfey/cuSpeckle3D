#include <string>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include "boolean_model.h"
#include <math.h> 
// #include "kernel.h"
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

//#define DEBUG


#define CUDA_CALL(x) do { if((x) != cudaSuccess) {printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)


// Mapping(displacement/deformation) function
__host__ __device__ __forceinline__ void mapping(float x_in, float y_in, float z_in,
                                                 float *x_out, float *y_out,float *z_out)
{
    // Identity function
    *x_out = x_in;
    *y_out = y_in;
    *z_out = z_in;
}

// Mapping(displacement/deformation) function
__host__ __device__ __forceinline__ void mappingStar(float x_in, float y_in, float z_in,int height, int width,int depth,
    float *x_out, float *y_out,float *z_out)
{
    const double PI= 3.14159265358979323846;
    //Minimum and maximum period of the wave
    float minPeriod=10;
    float maxPeriod=150;
    
    
    //Period of the wave such that it is equal to p_min at Z=0 and to p_max at Z=_z
    float period=minPeriod+(maxPeriod-minPeriod)*(z_in)/(depth-1);

    //Radial displacement = sine wave in the x-y plane
    float U_y;
    //U_y=0.5*cos((2*pi./p).*(Y-(dimension_y-1)/2-1));
    U_y=0.5*cos( (2.0*PI/period)*(y_in-(width-1)/2));

    *x_out = x_in ;// no displacement along x direction 
    *y_out = y_in + U_y;
    *z_out = z_in;// no displacement along z direction

    // *x_out = x_in + radialDisp * cos(theta);
    // *y_out = y_in + radialDisp * sin(theta);
    // *z_out = z_in;// no displacement along direction z
}


/*
delta estimation function
@dims : image dimensions
*/
void removeColumnDiff(int height, int width, int depth, float* meshVector_in, float *cutColumn_out, float* diffCutcolumn_out)
{
	//remove the column
	int page = 0;
	int index = 0;
	for (int row = 0; row < height; ++row)
	{
		for (int column = 1; column < width; ++column)
		{
			cutColumn_out[index] = meshVector_in[row * width + column + page * (width * height)];
			index++;
		}
	}
	
	for (int page = 1; page < depth; ++page)
	{
		for (int row = 0; row < height; ++row)
		{
			for (int column = 0; column < width; ++column)
			{
				cutColumn_out[index] = meshVector_in[row * width + column + page * (width * height)];
				index++;				
			}
		}
	}

	int newWidth = width * (depth - 1) + width - 1;

	#ifdef DEBUG
    std::cout << "Cut the first column:  \n";
	for (auto i = 0; i < height; ++i)
	{
		for (auto j = 0; j < newWidth; j++)
		{
			std::cout << cutColumn_out[i * newWidth + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << std::endl;
    #endif


    // diffrence squared (i+1 -i):
	index = 0;
	for (auto i = 0; i < height-1; ++i)
	{
		for (auto j = 0; j < newWidth; j++)
		{		
			diffCutcolumn_out[index] = (cutColumn_out[(i + 1) * newWidth + j] - cutColumn_out[i * newWidth + j]) * (cutColumn_out[(i + 1) * newWidth + j] - cutColumn_out[i * newWidth + j]);
			index++;
		}
		
	}
}

// Calculate diff(dX(2:end, : ), 1, 2). ^ 2 from Matlab
void removeRowDiff(int height, int width, int depth, float* meshVector_in, float* cutRow_out, float* diffCutRow_out)
{
	//remove the column
	int index = 0;
	for (int page = 0; page < depth; page++)
	{
		for (int row = 1; row < height; ++row)
		{
			for (int column = 0; column < width; ++column)
			{
				cutRow_out[index] = meshVector_in[row * width + column + page * (width * height)];
				index++;
			}            
		}
	}

	int newWidth = width* depth;
	#ifdef DEBUG
    std::cout << "Cut the first row:   \n" ;

	for (auto i = 0; i < height-1; ++i)
	{
		for (auto j = 0; j < newWidth; j++)
		{
			std::cout << cutRow_out[i * newWidth + j] << " ";
		}
		std::cout << std::endl;
	}
    #endif
	//diffrence squared (j+1 -j): 
	index = 0;
	for (auto i = 0; i < height - 1; ++i)
	{
		for (auto j = 0; j < newWidth-1; j++)
		{
			diffCutRow_out[index] = (cutRow_out[i * newWidth + j + 1] - cutRow_out[i * newWidth + j]) * (cutRow_out[i * newWidth + j + 1] - cutRow_out[i * newWidth + j]);
			index++;
		}
		
	}
}

float estimateDelta(float* diffXcolumn, float* diffXRow, float* diffYcolumn, float* diffYRow, float* diffZcolumn, float* diffZRow, float *sqF,int width, int height, int depth)
{
	
	int newWidth = width * depth -1;

	int index = 0;
	float MaxSqF = -INFINITY;
	for (auto i = 0; i < height - 1; i++)
	{
		for (auto j = 0; j < newWidth; ++j)
		{
			sqF[index]= diffXcolumn[i * newWidth + j] + diffXRow[i * newWidth + j] +
						diffYcolumn[i * newWidth + j] + diffYRow[i * newWidth + j] +
						diffZcolumn[i * newWidth + j] + diffZRow[i * newWidth + j];

            // save only the max value
			if (sqF[index] > MaxSqF)
			{
				MaxSqF = sqF[index];
			}
			index++;

		}

		//std::cout << std::endl;
	}
	return sqrt(MaxSqF);
}

 float estimate_delta(vec3D<int> dims)
{
    const int width  = dims.x;
    const int height = dims.y;
    const int depth  = dims.z;
    const int size = width * height * depth;
    float *dX, *dY, *dZ;
    dX = (float *)malloc(size * sizeof(float));
    dY = (float *)malloc(size * sizeof(float));
    dZ = (float *)malloc(size * sizeof(float));
    
    
    for (int page=0; page < depth; page++)
    {
        float z=page+1;
        for (int row = 0; row < height; ++row)
        {
            float y = row + 1; // fix the same "y" value in each row
            for (int column = 0; column < width; ++column)
            {
            float x = column + 1; // fix the same "x" value in each column

            float fx = 0, fy = 0, fz=0; // outputs of "fun"
              mapping(x, y, z,  &fx, &fy, &fz);
           // mappingStar(x,y,z,height,width,depth,&fx, &fy, &fz);
            dX[row* width + column + page*( width * height)] = fx - x;
            dY[row* width + column + page*( width * height)] = fy - y;
            dZ[row* width + column + page*( width * height)] = fz - z;
            }
        }
    }// This loop has been tested againt matlab loop and it gives the same results


    float* dXcutColumn ;
	float* dYcutColumn ;
	float* dZcutColumn ;
	dXcutColumn = (float*)malloc((size - height) * sizeof(float));
	dYcutColumn = (float*)malloc((size - height) * sizeof(float));
	dZcutColumn = (float*)malloc((size - height) * sizeof(float));


	float* dXcutRow;
	float* dYcutRow;
	float* dZcutRow;
	dXcutRow = (float*)malloc(((height - 1) * width * depth) * sizeof(float) );
	dYcutRow = (float*)malloc(((height - 1) * width * depth) * sizeof(float));
	dZcutRow = (float*)malloc(((height - 1) * width * depth) * sizeof(float));
	

    int newsize = (height - 1) * (width * (depth - 1) + width - 1);
	float* dXcutColumnDiff, * dYcutColumnDiff, * dZcutColumnDiff;
	float* dXcutRowDiff, * dYcutRowDiff, * dZcutRowDiff;

	dXcutColumnDiff = (float*)malloc(newsize * sizeof(float));
	dYcutColumnDiff = (float*)malloc(newsize * sizeof(float));
	dZcutColumnDiff = (float*)malloc(newsize * sizeof(float));
    
	dXcutRowDiff = (float*)malloc(newsize * sizeof(float));
	dYcutRowDiff = (float*)malloc(newsize * sizeof(float));
	dZcutRowDiff = (float*)malloc(newsize * sizeof(float));
    
    // Matlab equivalent ==>   diff(dX(:,2:end),1,1).^2 + diff(dX(2:end,:),1,2).^2 ..
    //                       + diff(dY(:,2:end),1,1).^2 + diff(dY(2:end,:),1,2).^2...
    //                       + diff(dZ(:.2:end),1,1).^2 + diff(dZ(2:end,:),1,2).^2;
    removeColumnDiff(height, width, depth, dX, dXcutColumn, dXcutColumnDiff);	
	removeColumnDiff(height, width, depth, dY, dYcutColumn, dYcutColumnDiff);	
	removeColumnDiff(height, width, depth, dZ, dZcutColumn, dZcutColumnDiff);

	removeRowDiff(height, width, depth, dX, dXcutRow, dXcutRowDiff);
	removeRowDiff(height, width, depth, dY, dYcutRow, dYcutRowDiff);
	removeRowDiff(height, width, depth, dZ, dZcutRow, dZcutRowDiff);
	
	float delta=0;	
	float* sqF;
	sqF = (float*)malloc(newsize * sizeof(float));

	delta=estimateDelta(dXcutColumnDiff, dXcutRowDiff, dYcutColumnDiff, dYcutRowDiff, dZcutColumnDiff, dZcutRowDiff, sqF, width, height, depth);
	
    // restrict the search zone if delta is bigger than delta>width/2 or delta>height/2 
    if((delta>width/2) || (delta>height/2) )
    {
        delta=0.3*delta;
    }

    free(sqF);

	free(dX);
	free(dY);
	free(dZ);

	free(dXcutColumn);
	free(dYcutColumn);
	free(dZcutColumn);

	free(dXcutRow);
	free(dYcutRow);
	free(dZcutRow);

	free(dXcutColumnDiff);
	free(dYcutColumnDiff);
	free(dZcutColumnDiff);

	free(dXcutRowDiff);
	free(dYcutRowDiff);
	free(dZcutRowDiff);
    return (float)(delta);
}


/** From cuda_samples/MC_EstimatePiP
 * @brief Calculate the sum within the block
 * 
 * @param in 
 * @param cta 
 * @return __device__ 
 */

__device__ unsigned int reduce_sum(unsigned int in, cg::thread_block cta)
{
    extern __shared__ unsigned int sdata[];

    // Perform first level of reduction:
    // - Write to shared memory
    unsigned int ltid = threadIdx.x;

    sdata[ltid] = in;
    cg::sync(cta);

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2 ; s > 0 ; s >>= 1)
    {
        if (ltid < s)
        {
            sdata[ltid] += sdata[ltid + s];
        }

        cg::sync(cta);
    }

    return sdata[0];
}

__global__ void setup_kernel(unsigned int seed, curandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(seed, id, 0, &state[id]);
}

/**
 * @brief Initialize the PRNG by affecting the same seed to each thread but a different sequence
 * 
 * @param state PRNG states' array
 * @param samples Number of MC experiments to execute
 * @param result output results of sub-sommation of the calculated sample mean 
 * @param x pixel's x coordinate
 * @param y pixel's y coordinate
 * @param z pixel's z coordinate
 * @param shpere_arr "L" search space
 * @param rc_size size of "L"
 * @param sigma standard deviation of the PSF
 * @return void
*/

__global__ void compute_intensity_kernel_float(curandStatePhilox4_32_10_t *state,
                                        int samples,
                                        unsigned int *result, 
                                        int x, int y, int z, Random_sphere* sphere_arr,
                                         int rc_size, float sigma,int width, int height, int depth)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    unsigned int bid = blockIdx.x;

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    const int max_itr_per_thread = samples/(blockDim.x * gridDim.x);
    const int r_itr_per_thread = samples%(blockDim.x * gridDim.x);
    
    unsigned int intensity = 0;

    // what is float2, i will try rand_var.z and see what would the error be like, is there float3
    float4 rand_var;// OR double2

    Random_sphere sphere; // tmp sphere
    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = state[id];
    /* Run MC simulation */

    for(int i = 0; i < max_itr_per_thread; ++i)
    {
        rand_var = curand_normal4 (&localState);// it was curand_normal2//
        float fx = 0, fy = 0, fz = 0;
        mapping(x + rand_var.x * sigma, y + rand_var.y * sigma,z+rand_var.z * sigma,&fx, &fy, &fz);
        //mappingStar(x + rand_var.x * sigma, y + rand_var.y * sigma,z+rand_var.z * sigma, height, width, depth,&fx, &fy, &fz);
        //check if phi(x+Xm) belongs to one of the spheres
        for (unsigned int k = 0; k < rc_size; ++k)
        {
            sphere = sphere_arr[k];
            float d1 = sphere.x - fx;
            float d2 = sphere.y - fy;
            float d3 = sphere.z - fz;
            float dist = d1 * d1 + d2 * d2+ d3 * d3;
            if (dist < sphere.r)
            {
                ++intensity;
                break;
            }
        }
    }

    // carry on the ramaining simulations on the first 'r_itr_per_thread' threads (in case (float)'max_itr_per_thread'=not int)
    if (id < r_itr_per_thread)
    {
        rand_var = curand_normal4(&localState);
        float fx = 0, fy = 0, fz = 0;
        mapping(x + rand_var.x * sigma, y + rand_var.y * sigma,z+rand_var.z * sigma, &fx, &fy, &fz);
        //mappingStar(x + rand_var.x * sigma, y + rand_var.y * sigma,z+rand_var.z * sigma,height, width, depth, &fx, &fy, &fz);

        // check if phi(x+Xm) belongs to one of the spheres
        for (unsigned int k = 0; k < rc_size; ++k)
        {
            sphere = sphere_arr[k];
            float d1 = sphere.x  - fx;
            float d2 = sphere.y  - fy;
            float d3 = sphere.z  - fz;
            float dist = d1 * d1 + d2 * d2+ d3 * d3;
            if (dist < sphere.r)
            {
                ++intensity;
                break;
            }
        }
    }

    /* Copy state back to global memory */
    state[id] = localState;
    // /* Store results */
    // result[id] += intensity;

    // Reduce within the block
    intensity = reduce_sum(intensity, cta);

    // Store the result
    if (threadIdx.x == 0)
    {
        result[bid] = intensity;
    }
}

/**
 * @brief Render the reel volume from the Boolean model using MC integration method
 * 
 * @param speckle_matrix Output rendered speckle image matrix
 * @param Random_centers Input generated sphere centers from Boolean model
 * @param Random_radius Input generated sphere radii from Boolean model
 * @param RBound Input RBound 
 * @param number toatal number of generated spheres
 * @param seed PRNG seed
 * @param width volume width
 * @param height volume height
 * @param depth volume depth
 * @param sigma standard deviation of the PSF
 * @param alpha quantization error probability
 * @param nbit bit depth
 * @param gamma speckle contrast
 * @param N0 sample size to estimate NMC
 * @return int 
*/

int monte_carlo_estimation_cuda(float *speckle_matrix,
                                float *Random_centers,
                                float *Random_radius,
                                float *RBound, int number,
                                unsigned int seed,
                                int width, int height, int depth,
                                float alpha, int nbit, float gamma, int N0, float sigma)
{

    //----- cuda Threads/Blocks setup variables(preparation) ----- ///
    struct cudaDeviceProp     deviceProperties;
    unsigned int device = gpuGetMaxGflopsDeviceId();

    // Get device properties
    cudaGetDeviceProperties(&deviceProperties, device);

    // Determine how to divide the work between cores
    dim3 block;
    dim3 grid;
    block.x = (unsigned int)deviceProperties.maxThreadsDim[0];
    grid.x  = (unsigned int)deviceProperties.maxGridSize[0];

    // Aim to launch around ten or more times as many blocks as there
    // are multiprocessors on the target device.
    unsigned int blocksPerSM = 4; // check ==> https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html
    unsigned int numSMs      = deviceProperties.multiProcessorCount;

    // make sure to use maximum grid size(nb blocks per SM) 
    while (grid.x > 2 * blocksPerSM * numSMs)
    {
        grid.x >>= 1;
    }

    // define nb of Threads per block staticly
    block.x = 64;

    const unsigned int totalThreads = block.x * grid.x;

    // Print Threads info
    printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    printf("Nb Blocks = %d\n", grid.x);
    printf("Nb Threads Per Block = %d\n", block.x);
    printf("Total nb_threads = %d\n", totalThreads);
    // ------------------------------- ///

    // calculation variables
    unsigned int total;
    curandStatePhilox4_32_10_t *devPHILOXStates;
    unsigned int *devResults, *hostResults;

    // define a constant variable needed to calculate NMC
    const float cst_var = (float)2 / pi * gamma * gamma * pow(2, 2 * nbit) / (alpha * alpha);

    /* Allocate space for results on host */
    hostResults = (unsigned int *)calloc(grid.x, sizeof(unsigned int));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&devResults, grid.x * sizeof(unsigned int)));

    CUDA_CALL(cudaMalloc((void **)&devPHILOXStates, totalThreads * sizeof(curandStatePhilox4_32_10_t)));

    /* Allocate memory for RC & RR */
    Random_sphere *sphere_arr;
    CUDA_CALL(cudaMallocManaged((void **)&sphere_arr, number * sizeof(*sphere_arr)));

    /* Compacting boolean model parameters into one array of structures */
    Boolean_model_sphere *BM_sphere_arr;
    BM_sphere_arr = (Boolean_model_sphere *)calloc(number, sizeof(Boolean_model_sphere));

    Boolean_model_sphere tmp;
    for (int i = 0; i < number; ++i) // do the compaction
    {
        tmp.sphere.x = Random_centers[3 * i];
        tmp.sphere.y = Random_centers[3 * i + 1];
        tmp.sphere.z = Random_centers[3 * i + 2];

        tmp.sphere.r = Random_radius[i] * Random_radius[i];
        tmp.rbound = RBound[i];
        BM_sphere_arr[i] = tmp;
    }
    
    /* Setup prng states */
    setup_kernel<<<grid, block>>>(seed, devPHILOXStates);

    // utility var 
    int count = 0;
    float dist;

    // Monte Carlo estimation
    for (int x = 1; x < height + 1; ++x)
    {
        for (int y = 1; y < width + 1; ++y)
        {
            for (int z=1; z< depth + 1; ++z)
            {
            
            float fx = 0, fy = 0, fz=0;
            //printf("fx1 is  = %f \n", fx); 
            float d1, d2, d3;
            mapping(x, y, z, &fx, &fy, &fz);
            //mappingStar(x , y,z,height, width, depth,&fx, &fy, &fz);
            // calculate L(x,y) = Ind
            count = 0; // size of RR
            Random_sphere sphere;
            Boolean_model_sphere bm_sphere;
                for (int i = 0; i < number; ++i)
                {
                bm_sphere = BM_sphere_arr[i];
                sphere = bm_sphere.sphere;
                d1 = sphere.x - fx;
                d2 = sphere.y - fy;
                d3 = sphere.z - fz;

                 
                //printf("Distance 1 is  = %f \n", d1);                
                //printf("Distance 2 is  = %f \n", d2);                
                //printf("Distance 3 is  = %f \n", d3);
                 
                dist = d1 * d1 + d2 * d2+ d3 * d3;
                
                    if (dist <= bm_sphere.rbound)
                    {
                        sphere_arr[count] = sphere;
                        count++;
                    }
                }
             
            /* Set results to 0 */
            CUDA_CALL(cudaMemset(devResults, 0, grid.x * sizeof(unsigned int)));

            //Monte Carlo estimation with sample size = N0
            compute_intensity_kernel_float<<<grid, block, block.x *sizeof(unsigned int)>>>(devPHILOXStates, N0, devResults, x, y,z, sphere_arr, count, sigma, height, width,depth);
            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostResults, devResults, grid.x *sizeof(unsigned int), cudaMemcpyDeviceToHost));

            /* Finish sum on host */
            total = 0;
            for(int i = 0; i < grid.x; i++) 
            {
                total += hostResults[i];
            }
            
            float intensity = (float)total/N0;

            // Estimation of Monte Carlo sample size NMC
            int NMC;
            NMC = floor((float)2 / pi * gamma * gamma * (intensity - intensity * intensity) * pow(2, 2 * nbit) / (alpha * alpha)) - N0;

            if (NMC < 1)
            {
                  speckle_matrix[(x-1)* width + (y-1) + (z-1)*( width * height)] = 1 - intensity; // x, y & z start at 1 instead of 0
            }
            else
            {
                float res = (1 - intensity) * ((float)N0 / (N0 + NMC));

                /* Set results to 0 */
                CUDA_CALL(cudaMemset(devResults, 0, grid.x * sizeof(unsigned int)));

                //Monte Carlo estimation with sample size = NMC
                compute_intensity_kernel_float<<<grid, block, block.x *sizeof(unsigned int)>>>(devPHILOXStates, NMC, devResults, x, y,z, sphere_arr, count,sigma,height,width,depth);
                /* Copy device memory to host */
                CUDA_CALL(cudaMemcpy(hostResults, devResults, grid.x * sizeof(unsigned int), cudaMemcpyDeviceToHost));

                /* Finish sum on host */
                total = 0;
                for(int i = 0; i < grid.x; i++)
                {
                    total += hostResults[i];
                }                
                
                float intensity = (float)total/NMC;
                speckle_matrix[(x-1)* width + (y-1) + (z-1)*( width * height)] = res + (1 - intensity) * ((float)NMC / (N0 + NMC));
            }

            }

        }
    }

    // Cleanup
    CUDA_CALL(cudaFree(sphere_arr));
    CUDA_CALL(cudaFree(devPHILOXStates));
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    free(BM_sphere_arr);

    for (int i = 0; i < width * height * depth; ++i)
        speckle_matrix[i] = pow(2, nbit - 1) + (gamma * pow(2, nbit) * (speckle_matrix[i] - 0.5));

    printf("*MC estimation CUDA test PASSED*\n");
    return EXIT_SUCCESS;
}