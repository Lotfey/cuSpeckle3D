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
		○ clone samples folder to this directory: usr/local/cuda "$ sudo git clone https://github.com/Lotfey/samples.git"

Declaration of mapping function (deformation):
	+ To generate reference image/volume use the identity function inside inside mapping fuction(check examples/example_id.txt)	
	save your files and rebuild your project then run command to gennerate your reference image/volume
	
	+ To gernerate deformed image/volume, you should define your deformation field then past your code 
	inside mapping function. In this project I used star deformation field function (check examples/example_star.txt)
	save your files and rebuild your project then run command to gennerate your deformed image/volume
	

Compilation & build:
	Run "make" command
	Verify that the compilation is successful and you can run "./cuSpeckle3D" on the terminal that returns the help of the program
	
Run Simple command: 
	./cuSpeckle3D img_out.png -width 100 -height 100 -depth 1
    or
    	./cuSpeckle3D vol_out -width 100 -height 100 -depth 100
	
Check the output image/ volume
