Pre-requisites:
	• A Linux operating system (tested on Ubuntu)
	• A NVIDIA GPU with a compute capability >= 6.0 (tested on CUDA 10.x and 11.x)
	
Installation steps:
	Install CUDA library (insure compute capability > …)
	Follow the official guide in https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#introduction
	Install dependencies
		○ sudo apt-get install libboost-all-dev
		○ sudo apt-get install libtiff-dev
		○ sudo apt-get install libpng-dev

Declaration of mapping function (deformation):
	replace the code inside "func" in file "speckle_generator_main.cpp" line 27 and inside "Disp" "operator" in file "MC_estimation_cuda.cu" line 32 
	with the same code to define the deesired mapping function to use (refere to the given examples)

Compilation & build:
	Run "make" command (optional name) ==> make TARGET="cuda_program"
	Verify that the compilation is successful and you can run "./cuSpeckle" on the terminal that returns the help of the program
	
Run Simple command: 
	./cuSpeckle3D img_out.png -width 100 -height 100 -depth 1
    or
    ./cuSpeckle3D vol_out -width 100 -height 100 -depth 100
	
Check the output image/ volume
