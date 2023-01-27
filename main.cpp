#include <iostream>
#include <cstdio>
#include <cuda_runtime_api.h>


bool checkGPUAvailable() {
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    //std::cout << "count of cuda devices: " << device_count << std::endl;
    if (gpuDevice > device_count) {
        std::cout << "Error: GPU device number is greater than the number of devices!" <<
            "Perhaps a CUDA-capable GPU is not installed?" << std::endl;
        return false;
    }
    else {
        return true;
    }
}
/**
 * C main function.
 */
int main(int argc, char* argv[]) {
    bool gpuAvailable = checkGPUAvailable();
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, gpuDevice);

    // audio playback

    //CPU loadin




    return true;
}
