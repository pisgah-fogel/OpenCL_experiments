#define CL_HPP_TARGET_OPENCL_VERSION 210

#include <iostream>
#include <CL/cl2.hpp>

void cpu_add_a_b() {
	int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
	int C[10];
	for (size_t i = 0; i<10; i++) {
		C[i] = A[i] + B[i];
	}
	std::cout<<"CPU a+b = ";
	for (size_t i = 0; i<10; i++) {
		std::cout<<C[i]<<" ";
	}
	std::cout<<std::endl;
}

void gpu_add_a_b() {
	//get all platforms (drivers)
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if(all_platforms.size()==0){
		std::cout<<" No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform=all_platforms[0];
	std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

	//get default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if(all_devices.size()==0){
		std::cout<<" No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device=all_devices[0];
	std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";


	cl::Context context({default_device});

	cl::Program::Sources sources;

	// kernel calculates for each element C=A+B
	std::string kernel_code=
		"   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
		"       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
		"   }                                                                               ";
	sources.push_back({kernel_code.c_str(),kernel_code.length()});

	cl::Program program(context,sources);
	if(program.build({default_device})!=CL_SUCCESS){
		std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
		exit(1);
	}


	// create buffers on the device
	cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*10);
	cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*10);
	cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*10);

	int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

	//create queue to which we will push commands for the device.
	cl::CommandQueue queue(context,default_device);

	//write arrays A and B to the device
	queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*10,A);
	queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int)*10,B);



	//alternative way to run the kernel
	cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
	  kernel_add.setArg(0,buffer_A);
	  kernel_add.setArg(1,buffer_B);
	  kernel_add.setArg(2,buffer_C);
	  queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange);
	  queue.finish();

	int C[10];
	//read result C from the device to array C
	queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*10,C);

	std::cout<<"GPU a+b = ";
	for (size_t i = 0; i<10; i++) {
		std::cout<<C[i]<<" ";
	}
	std::cout<<std::endl;

}

void cpu_sum() {
	unsigned int sum = 0;
	for (unsigned int i = 0; i < 1024*1024*128; i++)
		sum += 1;
	std::cout<<"CPU sum: "<<sum<<std::endl;
}

void gpu_sum_1() {
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if(all_platforms.size()==0){
		std::cout<<" No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform=all_platforms[0];
	std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if(all_devices.size()==0){
		std::cout<<" No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device=all_devices[0];
	std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";


	cl::Context context({default_device});

	cl::Program::Sources sources;

	std::string kernel_code=
		    "kernel void AtomicSum(global int* sum){"
        "local int tmpSum[4];"
        "if(get_local_id(0)<4){"
            "tmpSum[get_local_id(0)]=0;"
       " }"
        "barrier(CLK_LOCAL_MEM_FENCE);"
        "atomic_add(&tmpSum[get_global_id(0)%4],1);"
        "barrier(CLK_LOCAL_MEM_FENCE);"
        "if(get_local_id(0)==(get_local_size(0)-1)){"
            "atomic_add(sum,tmpSum[0]+tmpSum[1]+tmpSum[2]+tmpSum[3]);"
        "}"
    "}";
	sources.push_back({kernel_code.c_str(),kernel_code.length()});

	cl::Program program(context,sources);
	if(program.build({default_device})!=CL_SUCCESS){
		std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
		exit(1);
	}
    int sum=0;
    cl::Buffer bufferSum = cl::Buffer(context, CL_MEM_READ_WRITE, 1 * sizeof(int));
	cl::CommandQueue queue(context,default_device);
    queue.enqueueWriteBuffer(bufferSum, CL_TRUE, 0, 1 * sizeof(int), &sum);
    cl::Kernel kernel=cl::Kernel(program, "AtomicSum");
    kernel.setArg(0,bufferSum);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1024*1024*128), cl::NullRange);
    queue.finish();
     
    queue.enqueueReadBuffer(bufferSum,CL_TRUE,0,1 * sizeof(int),&sum);
    std::cout << "GPU (1) Sum: " << sum << "\n";
}

void gpu_sum_2() {
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if(all_platforms.size()==0){
		std::cout<<" No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform=all_platforms[0];
	std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if(all_devices.size()==0){
		std::cout<<" No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device=all_devices[0];
	std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

	cl::Context context({default_device});

	cl::Program::Sources sources;

	std::string kernel_code=
		"kernel void AtomicSum(global int* sum){"
		"atomic_add(sum,1);"
		"}";
	sources.push_back({kernel_code.c_str(),kernel_code.length()});

	cl::Program program(context,sources);
	if(program.build({default_device})!=CL_SUCCESS){
		std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
		exit(1);
	}

	cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int));
	int sum = 0;
	cl::CommandQueue queue(context,default_device);

	//write variable to the device
	queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int),&sum);

	// runthe kernel
	cl::Kernel kernel_add=cl::Kernel(program,"AtomicSum");
	  kernel_add.setArg(0,buffer_A);
	  queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(1024*1024*128),cl::NullRange);
	  queue.finish();

	queue.enqueueReadBuffer(buffer_A,CL_TRUE,0,sizeof(int),&sum);

	std::cout<<"GPU2 sum: "<<sum<<std::endl;

}

void benchmark(void (*foo)(void), size_t repeat) {
	auto begin = std::chrono::high_resolution_clock::now();
	for(size_t i = 0; i < repeat; i++)
	{
		(*foo)();
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
	double t = duration / repeat;
	std::cout << duration << "ns total time, average ("<<repeat<<") : " << t << "ns (";
	std::cout << t/1000 << "us, " << t/1000000 << "ms, " << t/1000000000 << "s)." << std::endl;
}

int main(){
	benchmark(&gpu_sum_1, 1);

	benchmark(&gpu_sum_2, 1);

	benchmark(&cpu_sum, 1);

	benchmark(&gpu_add_a_b, 100);

	benchmark(&cpu_add_a_b, 100);

	return 0;
}
