#include <iostream>
#include <cstdio>
#include <math.h>
#include <bit>
#include <chrono>
#include <cuda_runtime_api.h>

#include "AudioFile.h"
#include "gpufft.cuh"
#include "fft.hpp"

#include <Windows.h>
#pragma comment(lib, "Winmm.lib")
#include <mmsystem.h>

#include "portaudio.h"

bool checkGPUAvailable() {
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "count of cuda devices: " << device_count << std::endl;
    if (gpuDevice > device_count) {
        std::cout << "Error: GPU device number is greater than the number of devices!" <<
            "Perhaps a CUDA-capable GPU is not installed?" << std::endl;
        return false;
    }
    else {
        return true;
    }
}

void naiveConvolution(std::vector<double> &a, std::vector<double> &b, std::vector<double> &out) {
    out.resize(a.size() + b.size() - 1);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            out.at(i + j) += a.at(i) * b.at(j);
        }
    }
}

void fftConvolution(const std::vector<double> &dry, const std::vector<double> &ir, std::vector<double> &out, double wetGain) {
    size_t padded = std::bit_ceil(ir.size() + dry.size() - 1);

    // TODO: less clumsy way to handle this conversion?
    std::vector<std::complex<double>> complexDry;
    complexDry.reserve(padded); 
    std::transform(dry.cbegin(), dry.cend(), std::back_inserter(complexDry),
        [](double r) { return std::complex<double>(r); });
    complexDry.resize(padded, std::complex(0.));

    std::vector<std::complex<double>> complexIR;
    complexIR.reserve(padded);
    std::transform(ir.cbegin(), ir.cend(), std::back_inserter(complexIR),
        [](double r) { return std::complex<double>(r); });
    complexIR.resize(padded, std::complex(0.));

    fft(complexIR);
    fft(complexDry);

    // Pointwise product (assume cDry > cIR length)
    for (size_t i = 0; i < complexDry.size(); ++i) {
        complexDry.at(i) = complexDry.at(i) * complexIR.at(i);
    }

    ifft(complexDry);

    out.reserve(padded);
    // leave some of the mixing and mastery to user
    for (size_t i = 0; i < dry.size(); ++i) { // i < complexDry.size(); ++i) {
        double val = wetGain * complexDry.at(i).real();
        if (i < dry.size()) {
            val += dry[i];
        }
        // THINK below as alternative parametrization?
        // base + wet_mix_amount * (gain * real - base)

        out.push_back(val);
    }
}

int main(int argc, char* argv[]) {
    AudioFile<double> ir, dry;
    AudioFile<double>::AudioBuffer buffer;
    int channel = 0;
    double WET_GAIN = 0.155f; //TODO add a thing that decreases dry appropriately as well?
    std::string dryPath = "./samples/test_audio.wav";
    std::string irPath = "./samples/ftc_ir.wav";

    std::string outputNaive = "./samples/naiveConvolved.wav";
    std::string outputPath = "./samples/convolved.wav";

    buffer.resize(1);
    ir.load(irPath);
    dry.load(dryPath);

    std::cout << "Reading IR file: " << irPath << std::endl;;
    std::cout << "    Bit Depth : " << ir.getBitDepth() << std::endl;
    std::cout << "    Sampling Rate: " << ir.getSampleRate() << std::endl;
    std::cout << "    Num Channels: " << ir.getNumChannels() << std::endl;
    std::cout << "    Length in Seconds: " << ir.getLengthInSeconds() << std::endl;
    //std::cout << "Sample count: " << ir.samples[channel].size() << std::endl;
    std::cout << std::endl;
    std::cout << "Reading dry sound file: " << dryPath << std::endl;;
    std::cout << "    Bit Depth: " << dry.getBitDepth() << std::endl;
    std::cout << "    Sample Rate: " << dry.getSampleRate() << std::endl;
    std::cout << "    Num Channels : " << dry.getNumChannels() << std::endl;
    std::cout << "    Length in Seconds: " << dry.getLengthInSeconds() << std::endl;
    //std::cout << "Sample count: " << dry.samples[channel].size() << std::endl;

    // GPU Path
    //bool gpuAvailable = checkGPUAvailable(); // TODO errors if no GPU available
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, gpuDevice);

    // Main operation
    auto start = std::chrono::high_resolution_clock::now();
    fftConvolution(dry.samples[channel], ir.samples[channel], buffer[channel], WET_GAIN);
    //gpuFFTConvolution(dry.samples[channel], ir.samples[channel], buffer[channel], WET_GAIN);
    //olaFFTConv(dry.samples[channel], ir.samples[channel], buffer[channel], WET_GAIN, 4096);
    auto stop = std::chrono::high_resolution_clock::now();
    // End main operation

    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Calculation duration: " << duration.count() << " microseconds" << std::endl;

    AudioFile<double> convolved;
    convolved.setBitDepth(dry.getBitDepth());
    convolved.setSampleRate(dry.getSampleRate());
    convolved.setNumChannels(dry.getNumChannels());
    convolved.setAudioBuffer(buffer);
    convolved.save(outputPath, AudioFileFormat::Wave);

    PlaySound(outputPath.c_str(), NULL, SND_FILENAME);

    return 0;
}
