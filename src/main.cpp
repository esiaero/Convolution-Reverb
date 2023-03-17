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

// Recursive, in-place
//void fft(std::vector<std::complex<double>>& signal) {
//    using namespace std::complex_literals;
//
//    if (signal.size() <= 1) {
//        return;
//    }
//    else if ((signal.size() & (signal.size() - 1)) != 0) {
//        std::cout << "Size was " << signal.size() << ", needs to be a pow2" << std::endl;
//    }
//    else {
//        std::vector<std::complex<double>> even, odd;
//        even.reserve(signal.size() / 2);
//        odd.reserve(signal.size() / 2);
//        for (size_t i = 0; i < signal.size(); ) {
//            even.push_back(signal.at(i++));
//            odd.push_back(signal.at(i++));
//        }
//        fft(even);
//        fft(odd);
//
//        for (std::size_t i = 0; i < signal.size() / 2; ++i) {
//            std::complex<double> val = std::exp(-2. * 1i * M_PI * (double)i / (double)signal.size()) * odd.at(i);
//
//            signal.at(i) = even.at(i) + val;
//            signal.at(i + signal.size() / 2) = even.at(i) - val;
//        }
//    }
//}

// Breadth-first implementation, O(1) storage, ~ 1/2 runtime of recusrive
void fft(std::vector<std::complex<double>>& signal) {
    // Bit reverse permutation... kind of scuffed
    // http://graphics.stanford.edu/~seander/bithacks.html
    int log2len = (int)(round(std::log2(signal.size())));
    for (uint32_t a = 0; a < signal.size(); ++a) {
        uint32_t b = a;
        // Swap in groups of 1, 2, 4, 8, 16...
        b = (((b & 0XAAAAAAAA) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0XCCCCCCCC) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0XF0F0F0F0) >> 4) | ((b & 0x0F0F0F0F) << 4));
        b = (((b & 0xFF00FF00) >> 8) | ((b & 0x00FF00FF) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - log2len);

        if (b > a) {
            std::iter_swap(signal.begin() + a, signal.begin() + b);
        }
    }

    for (int m = 2; m <= signal.size(); m <<= 1) {
        for (int k = 0; k < signal.size(); k += m) {
            for (int j = 0; j < m / 2; ++j) {
                std::complex<double> even = signal.at(k + j);
                std::complex<double> odd = signal.at(k + j + m / 2);
                
                double term = -2. * M_PI * (double)j / (double)m;
                std::complex<double> val(cos(term), sin(term));
                val *= odd;
                // Below is minutely slower? Uncertain
                // using namespace std::complex_literals;
                // std::complex<double> val = std::exp(-2. * 1i * M_PI * (double)j / (double)m) * odd;

                signal.at(k + j) = even + val;
                signal.at(k + j + m / 2) = even - val;
            }
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

void fftConvolution(std::vector<double> &dry, std::vector<double> &ir, std::vector<double> &out, double wetGain) {
    size_t padded = std::bit_ceil(ir.size() + dry.size() - 1);
    out.resize(padded);

    // TODO: less clumsy way to handle this conversion? might need to change audio library
    // reinterpret cast may be possible
    std::vector<std::complex<double>> complexDry;
    complexDry.reserve(padded);
    std::transform(dry.cbegin(), dry.cend(), std::back_inserter(complexDry),
        [](double r) { return std::complex<double>(r); });
    for (size_t i = dry.size(); i < padded; ++i) {
        complexDry.push_back(std::complex(0.));
    }

    std::vector<std::complex<double>> complexIR;
    complexIR.reserve(padded);
    std::transform(ir.cbegin(), ir.cend(), std::back_inserter(complexIR),
        [](double r) { return std::complex<double>(r); });
    for (size_t i = ir.size(); i < padded; ++i) {
        complexIR.push_back(std::complex(0.));
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
    AudioFile<double>::AudioBuffer buffer;
    int channel = 0;
    double WET_GAIN = 0.115f;
    std::string dryPath = "./samples/aperture_dry.wav";
    std::string irPath = "./samples/dales_ir.wav";

    std::string outputNaive = "./samples/naiveConvolved.wav";
    std::string outputPath = "./samples/convolved.wav";

    buffer.resize(1);
    ir.load(irPath);
    dry.load(dryPath);

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


    //bool gpuAvailable = checkGPUAvailable();
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, gpuDevice);

    // Main operation
    auto start = std::chrono::high_resolution_clock::now();
    fftConvolution(dry.samples[channel], ir.samples[channel], buffer[channel], WET_GAIN);
    auto stop = std::chrono::high_resolution_clock::now();
    // End main operation


    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Duration: " << duration.count() << " microseconds" << std::endl;

    AudioFile<double> convolved;
    convolved.setBitDepth(dry.getBitDepth());
    convolved.setSampleRate(dry.getSampleRate());
    convolved.setNumChannels(dry.getNumChannels());
    convolved.setAudioBuffer(buffer);
    convolved.save(outputPath, AudioFileFormat::Wave);

    return 0;
}
