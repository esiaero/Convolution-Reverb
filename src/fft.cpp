#define _USE_MATH_DEFINES
#include "fft.hpp"

void fft(std::vector<std::complex<float>>& signal) {
    // Bit reverse permutation, roughly: http://graphics.stanford.edu/~seander/bithacks.html
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
                std::complex<float> even = signal.at(k + j);
                std::complex<float> odd = signal.at(k + j + m / 2);

                float term = -2. * M_PI * (float)j / (float)m;
                std::complex<float> val(cos(term), sin(term));
                val *= odd;
                // Below minutely slower? Uncertain
                // using namespace std::complex_literals;
                // std::complex<double> val = std::exp(-2. * 1i * M_PI * (double)j / (double)m) * odd;

                signal.at(k + j) = even + val;
                signal.at(k + j + m / 2) = even - val;
            }
        }
    }
}

void ifft(std::vector<std::complex<float>>& signal) {
    for (auto& v : signal) {
        v = std::conj(v);
    }
    fft(signal);
    for (auto& v : signal) {
        v = std::conj(v) / (float)signal.size(); // Scaling N
    }
}