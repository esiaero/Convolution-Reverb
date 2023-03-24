#include <device_launch_parameters.h>
#include <algorithm>
#include <complex>
#define _USE_MATH_DEFINES
#include <vector>
#include <cuda/std/complex>
#include <iostream>

#include "gpufft.cuh"
#include "fft.hpp"

constexpr unsigned blockSize = 64;

__global__ void bitReversePermute(cuda::std::complex<double> *signal, int log2len) {
    uint32_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t b = index;
    b = (((b & 0XAAAAAAAA) >> 1) | ((b & 0x55555555) << 1));
    b = (((b & 0XCCCCCCCC) >> 2) | ((b & 0x33333333) << 2));
    b = (((b & 0XF0F0F0F0) >> 4) | ((b & 0x0F0F0F0F) << 4));
    b = (((b & 0xFF00FF00) >> 8) | ((b & 0x00FF00FF) << 8));
    b = ((b >> 16) | (b << 16)) >> (32 - log2len);

    if (b > index) {
        cuda::std::complex<double> tmp = signal[index];
        signal[index] = signal[b];
        signal[b] = tmp;
    }
}

__global__ void fftHelperKernel(cuda::std::complex<double>* signal, int m, int k, int size) {
    uint32_t j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k + j + m / 2 < size && j < m / 2) {
        cuda::std::complex<double> even = signal[k + j];
        cuda::std::complex<double> odd = signal[k + j + m / 2];

        double term = -2. * M_PI * (double)j / (double)m;
        cuda::std::complex<double> val(cos(term), sin(term));
        val *= odd;
        signal[k + j] = even + val;
        signal[k + j + m / 2] = even - val;
    }
}

__global__ void pointwiseKernel(cuda::std::complex<double>* a, cuda::std::complex<double>* b) {
    uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    a[i] = a[i] * b[i];
}

__global__ void conjKernel(cuda::std::complex<double>* signal, double scale) {
    uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    signal[i] = cuda::std::conj(signal[i]) / scale;
}

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(size_t x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

// GPU convolution with Cooley-Tukey FFT.
// This is very unoptimized! and barely exploits GPU parallelism.
void gpuFFTConvolution(std::vector<double>& dry, std::vector<double>& ir, std::vector<double>& out, double wetGain)
{
    size_t padded = 1 << ilog2ceil(dry.size() + ir.size() - 1); // bit_ceil(dry.size() + ir.size() - 1);
    // clumsy refactoring // 
    std::vector<std::complex<double>> complexDry;
    complexDry.reserve(padded);
    std::transform(dry.cbegin(), dry.cend(), std::back_inserter(complexDry),
        [](double r) { return std::complex<double>(r); });
    complexDry.resize(padded, std::complex<double>(0.));

    std::vector<std::complex<double>> complexIR;
    complexIR.reserve(padded);
    std::transform(ir.cbegin(), ir.cend(), std::back_inserter(complexIR),
        [](double r) { return std::complex<double>(r); });
    complexIR.resize(padded, std::complex<double>(0.));
    // end clumsy refactoring //
    
    // todo? consider if cudaMemset can handle padding from above i.e. cudaMemset( , 0, ).
    cuda::std::complex<double>* dev_dry = 0;
    cuda::std::complex<double>* dev_ir = 0;

    cudaMalloc((void**)&dev_dry, padded * sizeof(std::complex<double>));
    cudaMalloc((void**)&dev_ir, padded * sizeof(std::complex<double>));
    cudaMemcpy(dev_dry, complexDry.data(), padded * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ir, complexIR.data(), padded * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

    dim3 fullBlocksPerGrid = (padded + blockSize - 1) / blockSize;

    bitReversePermute<<<fullBlocksPerGrid, blockSize>>>(dev_dry, (int)log2f(padded));
    bitReversePermute<<<fullBlocksPerGrid, blockSize>>>(dev_ir, (int)log2f(padded));
    for (int m = 2; m <= padded; m <<= 1) {
        for (int k = 0; k < padded; k += m) {
            dim3 blocksPerGrid = (m / 2 + blockSize - 1) / blockSize;
            fftHelperKernel<<<blocksPerGrid, blockSize>>>(dev_dry, m, k, padded);
            fftHelperKernel<<<blocksPerGrid, blockSize>>>(dev_ir, m, k, padded);
        }
    }

    pointwiseKernel<<<fullBlocksPerGrid, blockSize>>>(dev_dry, dev_ir);

    conjKernel<<<fullBlocksPerGrid, blockSize>>>(dev_dry, 1); //no scaling on this
    bitReversePermute<<<fullBlocksPerGrid, blockSize>>>(dev_dry, (int)log2f(padded));
    for (int m = 2; m <= padded; m <<= 1) {
        for (int k = 0; k < padded; k += m) {
            dim3 blocksPerGrid = (m / 2 + blockSize - 1) / blockSize;
            fftHelperKernel<<<blocksPerGrid, blockSize>>>(dev_dry, m, k, padded);
        }
    }
    conjKernel<<<fullBlocksPerGrid, blockSize>>>(dev_dry, padded);
    
    std::complex<double> *host_out = new std::complex<double>[padded];
    cudaMemcpy(host_out, dev_dry, padded * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

    out.reserve(padded);
    for (size_t i = 0; i < padded; ++i) {
        double val = wetGain * host_out[i].real();
        if (i < dry.size()) {
            val += dry[i];
        }
        out.push_back(val);
    }
    free(host_out);

Error:
    cudaFree(dev_dry);
    cudaFree(dev_ir);
    //return cudaStatus;
}
