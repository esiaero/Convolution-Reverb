#include <iostream>
#include <cstdio>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex>

#include <cuda_runtime_api.h>
#include "AudioFile.h"

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

// A simple function to compute the DFT. Unsurprisingly inefficient: O(N^2)
std::vector<std::complex<double>> dft(std::vector<std::complex<double>> signal) {
    using namespace std::complex_literals;
    std::vector<std::complex<double>> out;
    out.reserve(signal.size());

    for (std::size_t i = 0; i < signal.size(); ++i) {
        std::complex<double> sum = 0;
        for (std::size_t j = 0; j < signal.size(); ++j) {
            sum += signal.at(j) * std::exp(-2. * 1i * M_PI * (double)i * (double)j / (double)signal.size());
        }
        out.push_back(sum);
    }
    return out;
}

std::vector<std::complex<double>> fftRecursive(std::vector<std::complex<double>> signal) {
    using namespace std::complex_literals;
    std::vector<std::complex<double>> out;
    out.reserve(signal.size());

    if ((signal.size() & (signal.size() - 1)) != 0) { // not pow2
        std::cout << "Size was " << signal.size() << ", needs to be a pow2" << std::endl;
    }
    else if (signal.size() <= 2) {
        return dft(signal);
    }
    else {
        std::vector<std::complex<double>> even, odd;
        even.reserve(signal.size());
        odd.reserve(signal.size());
        for (std::size_t i = 0; i < signal.size(); ) {
            even.push_back(signal.at(i++));
            odd.push_back(signal.at(i++));
        }
        even = fftRecursive(even);
        odd = fftRecursive(odd);

        for (std::size_t i = 0; i < signal.size(); ++i) {
            std::complex<double> val = std::exp(-2. * 1i * M_PI * (double)i / (double)signal.size());
            int idx = i % (signal.size() / 2);
            out.push_back(even.at(idx) + val * odd.at(idx));
        }
        // consider the below for in-place option
        //for (std::size_t i = 0; i < signal.size() / 2; ++i) {
        //    std::complex<double> val = std::exp(-2. * 1i * M_PI * (double)i / (double)signal.size()) * odd.at(i);
        //    signal.at(i) = even.at(i) + val;
        //    signal.at(i + signal.size() / 2) = even.at(i) - val;
        //}
    }
    return out;
}

int main(int argc, char* argv[]) {
    AudioFile<double> ir, dry;
    int channel = 0;

    ir.load("./samples/falkland_tennis_court_ir.wav");
    dry.load("./samples/aperture_dry.wav");
    //std::cout << "IR Bit Depth: " << ir.getBitDepth() << std::endl;
    //std::cout << "IR Sample Rate: " << ir.getSampleRate() << std::endl;
    //std::cout << "IR Num Channels: " << ir.getNumChannels() << std::endl;
    //std::cout << "IR Length in Seconds: " << ir.getLengthInSeconds() << std::endl;
    //std::cout << "Sample count: " << ir.samples[channel].size() << std::endl;

    // TODO: less clumsy way to handle this? may be change audio library
    std::vector<std::complex<double>> complexIR;
    complexIR.reserve(ir.samples[channel].size());
    std::transform(std::begin(ir.samples[channel]), std::end(ir.samples[channel]), std::begin(complexIR),
        [](double r) { return std::complex<double>(r); });

    std::vector<std::complex<double>> complexDry;
    complexDry.reserve(dry.samples[channel].size());
    std::transform(std::begin(dry.samples[channel]), std::end(dry.samples[channel]), std::begin(complexDry),
        [](double r) { return std::complex<double>(r); });
    

    fftRecursive(complexIR);
    fftRecursive(complexDry);

    AudioFile<double> convolved;

    //bool gpuAvailable = checkGPUAvailable();
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, gpuDevice);

    // audio playback

    //CPU loadin


    std::string outputFilePath = "./samples/convolved.wav"; // change this to somewhere useful for you
    convolved.save(outputFilePath, AudioFileFormat::Aiff);

    return 0;
}
