#include <iostream>
#include <cstdio>
#include <bit>
#include <chrono>

#include "fft.hpp"

#include "portaudio.h"
#include "AudioFile.h"

/*
 * This duplex audio setup is based from an example provided by PortAudio
 *
 * This program uses the PortAudio Portable Audio Library.
 * For more information see: http://www.portaudio.com
 * Copyright (c) 1999-2000 Ross Bencina and Phil Burk
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#define PA_SAMPLE_TYPE      paFloat32
constexpr int SAMPLE_RATE = 44100;
constexpr int FRAMES_PER_BUFFER = 131072;
constexpr float WET_GAIN = 0.155f; //TODO add a thing that decreases dry appropriately as well?

std::vector<std::complex<float>> complexDry;
std::vector<std::complex<float>> complexIR;
std::vector<float> olaBuffer;

static int callback(
    const void* inputBuffer,
    void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData)
{
    float* out = (float*)outputBuffer;
    const float* in = (const float*)inputBuffer;
    unsigned int i;
    (void)timeInfo; /* Prevent unused variable warnings. */
    (void)statusFlags;
    (void)userData;

    for (i = 0; i < complexDry.size(); ++i) {
        if (i < framesPerBuffer) {
            complexDry.at(i) = std::complex<float>(*in++);
        }
        else {
            complexDry.at(i) = 0;
        }
    }

    fft(complexDry);

    for (i = 0; i < complexDry.size(); ++i) { // dry vs ir length?
        complexDry.at(i) *= complexIR.at(i);
    }

    ifft(complexDry);

    in = (const float*)inputBuffer;
    double max = 0;
    for (i = 0; i < complexDry.size(); ++i) {
        if (i < framesPerBuffer) { 
            *out = *in++ + WET_GAIN * complexDry.at(i).real();

            // assume std::bit_ceil(framesPerBuffer + ir.size() - 1) - framesPerBuffer < framesPerBuffer
            // otherwise buffer needs to add to itself
            if (i < olaBuffer.size()) {
                *out += olaBuffer.at(i);
            }
            out++;
        }
        else {
            olaBuffer.at(i - framesPerBuffer) = WET_GAIN * complexDry.at(i).real();
        }
    }

    return paContinue;
}

int main(void)
{
    AudioFile<float> irFile;
    int channel = 0;
    std::string irPath = "./samples/dales_ir.wav";
    irFile.load(irPath);
    std::cout << "Reading IR file: " << irPath << std::endl;;
    std::cout << "    Sampling Rate: " << irFile.getSampleRate() << std::endl;
    std::cout << "    Num Channels: " << irFile.getNumChannels() << std::endl;
    std::cout << "    Sample Count: " << irFile.samples[channel].size() << std::endl;
    std::cout << "    Length (s): " << irFile.getLengthInSeconds() << std::endl;
    std::cout << std::endl;

    std::vector<float> ir = irFile.samples[channel];
    size_t padded = std::bit_ceil(ir.size() + FRAMES_PER_BUFFER - 1);
    complexIR.reserve(padded);
    std::transform(ir.cbegin(), ir.cend(), std::back_inserter(complexIR),
        [](float r) { return std::complex<float>(r); });
    complexIR.resize(padded, std::complex(0.f));
    fft(complexIR);

    complexDry.resize(padded, std::complex(0.f));

    olaBuffer.resize(padded - FRAMES_PER_BUFFER, 0.f);

    PaStream* stream;
    PaError err = Pa_Initialize();
    if (err != paNoError) goto error;

    PaStreamParameters inputParameters, outputParameters;

    inputParameters.device = Pa_GetDefaultInputDevice(); /* default input device */
    if (inputParameters.device == paNoDevice) {
        fprintf(stderr, "Error: No default input device.\n");
        goto error;
    }
    inputParameters.channelCount = 1;
    inputParameters.sampleFormat = PA_SAMPLE_TYPE;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultHighInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;

    outputParameters.device = Pa_GetDefaultOutputDevice(); /* default output device */
    if (outputParameters.device == paNoDevice) {
        fprintf(stderr, "Error: No default output device.\n");
        goto error;
    }
    outputParameters.channelCount = 1;
    outputParameters.sampleFormat = PA_SAMPLE_TYPE;
    outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultHighInputLatency;
    outputParameters.hostApiSpecificStreamInfo = NULL;

    err = Pa_OpenStream( // try defaultopen with 1 1
        &stream,
        &inputParameters,
        &outputParameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paNoFlag,
        callback,
        NULL);
    if (err != paNoError) goto error;

    err = Pa_StartStream(stream);
    if (err != paNoError) goto error;

    printf("Press ENTER to stop program.\n");
    getchar();
    err = Pa_CloseStream(stream);
    if (err != paNoError) goto error;

    Pa_Terminate();
    return 0;

error:
    Pa_Terminate();
    std::cerr << "PortAudio error occurred. Error message: " << Pa_GetErrorText(err) << std::endl;
    return -1;
}
