#include <iostream>
#include <cstdio>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex>
#include <bit>
#include <chrono>

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

void fft(std::vector<std::complex<double>>& signal) {
    using namespace std::complex_literals;

    if (signal.size() <= 1) {
        return;
    }
    else if ((signal.size() & (signal.size() - 1)) != 0) {
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
        fft(even);
        fft(odd);

        for (std::size_t i = 0; i < signal.size() / 2; ++i) {
            std::complex<double> val = std::exp(-2. * 1i * M_PI * (double)i / (double)signal.size()) * odd.at(i);

            signal.at(i) = even.at(i) + val;
            signal.at(i + signal.size() / 2) = even.at(i) - val;
        }
    }
}

void ifft(std::vector<std::complex<double>>& signal) {
    for (auto& v : signal) {
        v = std::conj(v);
    }
    fft(signal);
    for (auto& v : signal) {
        v = std::conj(v) / (double)signal.size(); // Scaling N
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

void fftConvolution(std::vector<double> &a, std::vector<double> &dry, std::vector<double> &out, double wetGain) {
    size_t padded = std::bit_ceil(a.size() + dry.size() - 1); 
    out.resize(padded);

    // TODO: less clumsy way to handle this conversion? might need to change audio library
    std::vector<std::complex<double>> complexIR;
    complexIR.reserve(padded);
    std::transform(a.cbegin(), a.cend(), std::back_inserter(complexIR),
        [](double r) { return std::complex<double>(r); });
    for (size_t i = a.size(); i < padded; ++i) {
        complexIR.push_back(std::complex(0.));
    }

    std::vector<std::complex<double>> complexDry;
    complexDry.reserve(padded);
    std::transform(dry.cbegin(), dry.cend(), std::back_inserter(complexDry),
        [](double r) { return std::complex<double>(r); });
    for (size_t i = dry.size(); i < padded; ++i) {
        complexDry.push_back(std::complex(0.));
    }

    fft(complexIR);
    fft(complexDry);

    // Pointwise product (assume cDry > cIR length)
    for (size_t i = 0; i < complexDry.size(); ++i) {
        complexDry.at(i) = complexDry.at(i) * complexIR.at(i);
    }

    ifft(complexDry);

    // leave some of the mixing and mastery to user
    for (size_t i = 0; i < complexDry.size(); ++i) {
        double base = 0;
        if (i < dry.size()) {
            base = dry[i];
        }
        // THINK below as alternative parametrization?
        // base + wet_mix_amount * (gain * real - base)

        double val = base + wetGain * complexDry.at(i).real();
        out.at(i) = val;
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

    AudioFile<double>::AudioBuffer buffer;
    buffer.resize(1);
    double WET_GAIN = 0.115f;

    auto start = std::chrono::high_resolution_clock::now();
    fftConvolution(ir.samples[channel], dry.samples[channel], buffer[channel], WET_GAIN);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Duration: " << duration.count() << " microseconds" << std::endl;

    convolved.setAudioBuffer(buffer);
    convolved.save(outputFilePath, AudioFileFormat::Wave);

    //bool gpuAvailable = checkGPUAvailable();
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, gpuDevice);
    return 0;
}
