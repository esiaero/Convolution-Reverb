#include <iostream>
#include <cstdio>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex>
#include <bit>

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

void convolution(std::vector<double> &a, std::vector<double> &b, std::vector<double> &out) {
    out.resize(a.size() + b.size() - 1);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            out.at(i + j) += a.at(i) * b.at(j);
        }
    }
}

/*
* A simple function to compute the DFT. Unsurprisingly inefficient : O(N ^ 2)
*/ 
//void dft(std::vector<std::complex<double>> &signal) {
//    using namespace std::complex_literals;
//
//    for (size_t i = 0; i < signal.size(); ++i) {
//        std::complex<double> sum = 0;
//        for (size_t j = 0; j < signal.size(); ++j) {
//            sum += signal.at(j) * std::exp(-2. * 1i * M_PI * (double)i * (double)j / (double)signal.size());
//        }
//        signal.at(i) = sum;
//    }
//}

void fftRecursive(std::vector<std::complex<double>> &signal) {
    using namespace std::complex_literals;

    if (signal.size() <= 1) {
        return;
    }
    else if ((signal.size() & (signal.size() - 1)) != 0) { // not pow2
        std::cout << "Size was " << signal.size() << ", needs to be a pow2" << std::endl;
    }
    else {
        std::vector<std::complex<double>> even, odd;
        even.reserve(signal.size() / 2);
        odd.reserve(signal.size() / 2);
        for (size_t i = 0; i < signal.size(); ) {
            even.push_back(signal.at(i++));
            odd.push_back(signal.at(i++));
        }
        fftRecursive(even);
        fftRecursive(odd);

        for (size_t i = 0; i < signal.size(); ++i) {
            std::complex<double> val = std::exp(-2. * 1i * M_PI * (double)i / (double)signal.size());
            int idx = i % (signal.size() / 2);
            signal.at(i) = even.at(idx) + val * odd.at(idx);
        }
        // consider the below for faster? double check
        //for (std::size_t i = 0; i < signal.size() / 2; ++i) {
        //    std::complex<double> val = std::exp(-2. * 1i * M_PI * (double)i / (double)signal.size()) * odd.at(i);

        //    signal.at(i) = even.at(i) + val;
        //    signal.at(i + signal.size() / 2) = even.at(i) - val;
        //}
    }
}

void ifft(std::vector<std::complex<double>>& signal) {
    for (auto& v : signal) {
        v = std::conj(v);
    }
    fftRecursive(signal);
    for (auto& v : signal) {
        v = std::conj(v) / (double)signal.size(); // Scaling N
    }
}

int main(int argc, char* argv[]) {
    AudioFile<double> ir, dry;
    int channel = 0;
    std::string outputNaive = "./samples/naiveConvolved.wav";
    std::string outputFilePath = "./samples/convolved.wav";

    ir.load("./samples/dales_ir.wav");
    dry.load("./samples/aperture_dry.wav");

    AudioFile<double> convolved;
    convolved.setBitDepth(dry.getBitDepth());
    convolved.setSampleRate(dry.getSampleRate());
    convolved.setNumChannels(dry.getNumChannels());

    //std::cout << "IR Bit Depth: " << ir.getBitDepth() << std::endl;
    //std::cout << "IR Sample Rate: " << ir.getSampleRate() << std::endl;
    //std::cout << "IR Num Channels: " << ir.getNumChannels() << std::endl;
    std::cout << "IR Length in Seconds: " << ir.getLengthInSeconds() << std::endl;
    //std::cout << "Sample count: " << ir.samples[channel].size() << std::endl;
    std::cout << std::endl;
    std::cout << "Dry Bit Depth: " << dry.getBitDepth() << std::endl;
    std::cout << "Dry Sample Rate: " << dry.getSampleRate() << std::endl;
    std::cout << "Dry Num Channels: " << dry.getNumChannels() << std::endl;
    std::cout << "Dry Length in Seconds: " << dry.getLengthInSeconds() << std::endl;
    std::cout << "Sample count: " << dry.samples[channel].size() << std::endl;

    /////// Naive convolution test
    //AudioFile<double>::AudioBuffer buffer;
    //buffer.resize(1);
    //convolution(ir.samples[channel], dry.samples[channel], buffer[channel]);
    //convolved.setAudioBuffer(buffer);
    //convolved.save(outputNaive, AudioFileFormat::Wave);
    //-----

    // TODO: less clumsy way to handle this? may be change audio library
    // zero pad for cooleytukey
    std::vector<std::complex<double>> complexIR;
    size_t padded = std::bit_ceil(ir.samples[channel].size() + dry.samples[channel].size() - 1); 
    complexIR.reserve(padded);
    std::transform(ir.samples[channel].cbegin(), ir.samples[channel].cend(), std::back_inserter(complexIR),
        [](double r) { return std::complex<double>(r); });
    for (size_t i = ir.samples[channel].size(); i < padded; ++i) {
        complexIR.push_back(std::complex(0.));
    }

    std::vector<std::complex<double>> complexDry;
    complexDry.reserve(padded);
    std::transform(dry.samples[channel].cbegin(), dry.samples[channel].cend(), std::back_inserter(complexDry),
        [](double r) { return std::complex<double>(r); });
    for (size_t i = dry.samples[channel].size(); i < padded; ++i) {
        complexDry.push_back(std::complex(0.));
    }

    fftRecursive(complexIR);
    fftRecursive(complexDry);

    // Pointwise product (assume cDry > cIR length)
    for (size_t i = 0; i < complexDry.size(); ++i) {
        complexDry.at(i) = complexDry.at(i) * complexIR.at(i);
    }

    ifft(complexDry);

    double WET_GAIN = 0.115f;
    // TODO: below needed to work because of agnostic scaling? leave some of the mixing and mastery to user.
    // even so, convolved audio seems to peak/distort incorrectly
    for (size_t i = 0; i < complexDry.size(); ++i) {
        double base = 0;
        if (i < dry.samples[channel].size()) {
            base = dry.samples[channel][i];
        }
        // THINK below as alternative parametrization?
        // base + wet_mix_amount * (gain * real - base)

        double val = base + WET_GAIN * complexDry.at(i).real();
        // val = complexDry.at(i).real(); // this sound very bad? y

        convolved.samples[channel].push_back(val);
    }

    //bool gpuAvailable = checkGPUAvailable();
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, gpuDevice);

    convolved.save(outputFilePath, AudioFileFormat::Wave);

    return 0;
}
