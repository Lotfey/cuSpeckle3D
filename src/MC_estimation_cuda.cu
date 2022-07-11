#include <string>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include "boolean_model.h"
// #include "kernel.h"

#include <helper_cuda.h>
#include <cuda_runtime.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)


// mapping(displacement/deformation) function
__host__ __device__ __forceinline__ void mapping(float x_in, float y_in, float z_in,
                                                float *x_out, float *y_out,float *z_out)
{
    // identity function
    *x_out = x_in;
    *y_out = y_in;
    *z_out = z_in;
}


/*
delta estimation function
@dims : image dimensions
*/
float estimate_delta(vec3D<int> dims)
{
    return 0;

}
/*
 float estimate_delta(vec3D<int> dims)
{
    const int width = dims.x;
    const int height = dims.y;
    const int depth = dims.z;
    const int size = width * height * depth;
    float *dX, *dY, *dZ;
    dX = (float *)malloc(size * sizeof(float));
    dY = (float *)malloc(size * sizeof(float));
    dZ = (float *)malloc(size * sizeof(float));
    
    
    for (int row = 0; row < height; ++row)
    {
        float y = row + 1; // fix the same "y" value in each row
        for (int column = 0; row < width; ++column)
            {
                float z = column + 1; // fix the same "z" value in each column
                for (int page = 0; page < depth; ++page)
                {
                float x = page + 1; // fix the same "x" value in each page

                float fx = 0, fy = 0, fz=0; // outputs of "fun"
                mapping(x, y,z,  &fx, &fy, &fz);
                dX[(column + row * width) * 3 +page] = fx - x;
                dY[(column + row * width) * 3 +page] = fy - y;
                dZ[(column + row * width) * 3 +page] = fx - z;
                }
            }
    }


    float *sqF;
    sqF = (float *)malloc((width - 1) * (height - 1) * (depth - 1) * sizeof(float));
    float max = -INFINITY;
    float sqF_r_c = 0;
    // Matlab equivalent ==>  diff(dX(:,2:end),1,1).^2 + diff(dY(:,2:end),1,1).^2 + diff(dX(2:end,:),1,2).^2 + diff(dY(2:end,:),1,2).^2;
    for (int row = 0; row < height - 1; ++row)
    {
        for (int column = 0; column < width - 1; ++column)
        {
            for (int page = 0; page < depth - 1; ++page)
            {
            sqF_r_c =  pow(dX[((column + 1) + row * width) * 3 + page] - dX[((column + 1) + (row + 1) * width) * 3 + page], 2);
            sqF_r_c += pow(dY[((column + 1) + row * width) * 3 + page] - dY[((column + 1) + (row + 1) * width) * 3 + page], 2);
            sqF_r_c += pow(dZ[((column + 1) + row * width) * 3 + page] - dZ[((column + 1) + (row + 1) * width) * 3 + page], 2);

            sqF_r_c += pow(dX[((column + 1) + row * width) * 3 + page] - dX[((column + 1) + (row + 1) * width) * 3 + page], 2);
            sqF_r_c += pow(dY[((column + 1) + row * width) * 3 + page] - dY[((column + 1) + (row + 1) * width) * 3 + page], 2);
            sqF_r_c += pow(dZ[((column + 1) + row * width) * 3 + page] - dZ[((column + 1) + (row + 1) * width) * 3 + page], 2);
                        
            // ((column + 1) + row * width)+page previous index equation " this is wrong "
                        //(column + 1)*3 + row * width*3+page
            sqF[(column + row * (width - 1)) * 3 +page] = sqF_r_c;
            max = (sqF[(column  + row * (width - 1)) * 3 + page] > max) ? sqF[(column  + row * (width - 1)) * 3 + page] : max; // calculate max
            }
        
        }
    }

    free(dX);
    free(dY);
    free(dZ);
    free(sqF);

    return (float)(sqrt(max));
}

*/

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

__global__ void compute_intensity_kernel_float(curandStatePhilox4_32_10_t *state,
                                        int samples,
                                        unsigned int *result, 
                                        int x, int y, int z, Random_sphere* sphere_arr, int rc_size)
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
        mapping(x + rand_var.x, y + rand_var.y,z+rand_var.z,
                 &fx, &fy, &fz);

        // check if phi(x+Xm) belongs to one of the spheres
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
        rand_var = curand_normal4(&localState);// same problem of 175 line code 

        float fx = 0, fy = 0, fz = 0;
        mapping(x + rand_var.x, y + rand_var.y,z+rand_var.z,
                 &fx, &fy, &fz);

        // check if phi(x+Xm) belongs to one of the spheres
        for (unsigned int k = 0; k < rc_size; ++k)
        {
            sphere = sphere_arr[k];
            float d1 = sphere.x  - fx;
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

int monte_carlo_estimation_cuda(float *speckle_matrix,
                                float *Random_centers,
                                float *Random_radius,
                                float *RBound, int number,
                                unsigned int seed,
                                int width, int height, int depth,
                                float alpha, int nbit, float gamma, int N0)
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
            float d1, d2, d3;
            mapping(x, y, z, &fx, &fy, &fz);
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

                dist = d1 * d1 + d2 * d2+ d3 * d3;
                dist=dist*dist;

                if (dist <= bm_sphere.rbound)
                {
                    sphere_arr[count] = sphere;
                    count++;
                }
            }

            /* Set results to 0 */
            CUDA_CALL(cudaMemset(devResults, 0, grid.x * sizeof(unsigned int)));

            //Monte Carlo estimation with sample size = N0
            compute_intensity_kernel_float<<<grid, block, block.x *sizeof(unsigned int)>>>(devPHILOXStates, N0, devResults, x, y,z, sphere_arr, count);
            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostResults, devResults, grid.x *
                sizeof(unsigned int), cudaMemcpyDeviceToHost));

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
                // x  + height *( y + depth*z)
                //(x - 1) + height*( (y - 1)+ depth*z)
                speckle_matrix[(x-1)* width + (y-1) + (z-1)*( width * height)] = 1 - intensity; // x & y start at 1 instead of 0
            }
            else
            {
                float res = (1 - intensity) * ((float)N0 / (N0 + NMC));

                /* Set results to 0 */
                CUDA_CALL(cudaMemset(devResults, 0, grid.x * sizeof(unsigned int)));

                //Monte Carlo estimation with sample size = NMC
                compute_intensity_kernel_float<<<grid, block, block.x *sizeof(unsigned int)>>>(devPHILOXStates, NMC, devResults, x, y,z, sphere_arr, count);
                /* Copy device memory to host */
                CUDA_CALL(cudaMemcpy(hostResults, devResults, grid.x *
                    sizeof(unsigned int), cudaMemcpyDeviceToHost));

                /* Finish sum on host */
                total = 0;
                for(int i = 0; i < grid.x; i++) {
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

    printf("^^^^ MC estimation CUDA test PASSED\n");

    return EXIT_SUCCESS;
}