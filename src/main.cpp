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

// A simple function to compute the DFT. Unsurprisingly inefficient: O(N^2)
void dft(std::vector<std::complex<double>> &signal) {
    using namespace std::complex_literals;

    for (std::size_t i = 0; i < signal.size(); ++i) {
        std::complex<double> sum = 0;
        for (std::size_t j = 0; j < signal.size(); ++j) {
            sum += signal.at(j) * std::exp(-2. * 1i * M_PI * (double)i * (double)j / (double)signal.size());
        }
        signal.at(i) = sum;
    }
}

void fftRecursive(std::vector<std::complex<double>> &signal) {
    using namespace std::complex_literals;

    if ((signal.size() & (signal.size() - 1)) != 0) { // not pow2
        std::cout << "Size was " << signal.size() << ", needs to be a pow2" << std::endl;
    }
    else if (signal.size() <= 2) {
        dft(signal);
    }
    else {
        std::vector<std::complex<double>> even, odd;
        even.reserve(signal.size());
        odd.reserve(signal.size());
        for (std::size_t i = 0; i < signal.size(); ) {
            even.push_back(signal.at(i++));
            odd.push_back(signal.at(i++));
        }
        fftRecursive(even);
        fftRecursive(odd);

        for (std::size_t i = 0; i < signal.size(); ++i) {
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
        v = std::conj(v) / (double)signal.size();
    }
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

    std::cout << "Dry Bit Depth: " << dry.getBitDepth() << std::endl;
    std::cout << "Dry Sample Rate: " << dry.getSampleRate() << std::endl;
    std::cout << "Dry Num Channels: " << dry.getNumChannels() << std::endl;
    std::cout << "Dry Length in Seconds: " << dry.getLengthInSeconds() << std::endl;
    std::cout << "Sample count: " << dry.samples[channel].size() << std::endl;

    // TODO: less clumsy way to handle this? may be change audio library
    // zero pad for cooleytukey
    std::vector<std::complex<double>> complexIR;
    size_t padded = std::bit_ceil(ir.samples[channel].size());
    complexIR.reserve(padded);
    std::transform(ir.samples[channel].cbegin(), ir.samples[channel].cend(), std::back_inserter(complexIR),
        [](double r) { return std::complex<double>(r); });
    for (size_t i = ir.samples[channel].size(); i < padded; ++i) {
        complexIR.push_back(std::complex(0.));
    }

    std::vector<std::complex<double>> complexDry;
    padded = std::bit_ceil(dry.samples[channel].size());
    complexDry.reserve(padded);
    std::transform(dry.samples[channel].cbegin(), dry.samples[channel].cend(), std::back_inserter(complexDry),
        [](double r) { return std::complex<double>(r); });
    for (size_t i = dry.samples[channel].size(); i < padded; ++i) {
        complexDry.push_back(std::complex(0.));
    }
    
    fftRecursive(complexIR);
    fftRecursive(complexDry);
    // Pointwise product
    for (size_t i = 0; i < complexDry.size(); ++i) {
        complexDry.at(i) *= complexIR.at(i % complexIR.size());
    }

    ifft(complexDry);


    AudioFile<double>::AudioBuffer buffer;
    buffer.resize(1);
    buffer[0].resize(complexDry.size());
    for (size_t i = 0; i < complexDry.size(); ++i) {
        buffer[channel][i] = complexDry.at(i).real();
    }

    AudioFile<double> convolved;
    convolved.setBitDepth(dry.getBitDepth());
    convolved.setSampleRate(dry.getSampleRate());
    convolved.setNumChannels(dry.getNumChannels());
    //bool ok = convolved.setAudioBuffer(buffer);
    //std::cout << ok << std::endl;
    for (size_t i = 0; i < complexDry.size(); ++i) {
        convolved.samples[channel].push_back(complexDry.at(i).real());
    }
    std::cout << convolved.samples[channel].size();

    //bool gpuAvailable = checkGPUAvailable();
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, gpuDevice);

    std::string outputFilePath = "./samples/convolved.wav"; // change this to somewhere useful for you
    convolved.save(outputFilePath, AudioFileFormat::Wave);

    return 0;
}
