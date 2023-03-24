#define _USE_MATH_DEFINES
#include "fft.hpp"

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
                std::complex<double> even = signal.at(k + j);
                std::complex<double> odd = signal.at(k + j + m / 2);

                double term = -2. * M_PI * (double)j / (double)m;
                std::complex<double> val(cos(term), sin(term));
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

void ifft(std::vector<std::complex<double>>& signal) {
    for (auto& v : signal) {
        v = std::conj(v);
    }
    fft(signal);
    for (auto& v : signal) {
        v = std::conj(v) / (double)signal.size(); // Scaling N
    }
}