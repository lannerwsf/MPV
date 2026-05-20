#include "FFTProcessor.h"
#include <fftw3.h>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

FFTProcessor::FFTProcessor(size_t bufferSize)
    : bufferSize(bufferSize), magnitudes(bufferSize / 2, 0.0f),
      bass(0.0f), mid(0.0f), treble(0.0f),
      bassAtt(0.0f), midAtt(0.0f), trebleAtt(0.0f) {
    // Allocate FFT input/output arrays
    fftInput = new float[bufferSize];
    fftOutput = new float[bufferSize];

    // Create FFTW plan
    fftPlan = fftwf_plan_r2r_1d(bufferSize, fftInput, fftOutput, FFTW_R2HC, FFTW_MEASURE);
    if (!fftPlan) {
        cerr << "Failed to create FFTW plan." << endl;
    }
}

FFTProcessor::~FFTProcessor() {
    // Destroy FFTW plan and free memory
    fftwf_destroy_plan(static_cast<fftwf_plan>(fftPlan));
    delete[] fftInput;
    delete[] fftOutput;
}

void FFTProcessor::computeFFT(const vector<float>& audioData) {
    if (audioData.size() < bufferSize) {
        cerr << "Audio data size is smaller than the buffer size." << endl;
        return;
    }

    // Copy audio data to fftInput
    for (size_t i = 0; i < bufferSize; ++i) {
        fftInput[i] = audioData[i];
    }

    // Execute FFT
    fftwf_execute(static_cast<fftwf_plan>(fftPlan));

    // Compute magnitudes from FFT output
    size_t numBins = bufferSize / 2;
    for (size_t i = 0; i < numBins; ++i) {
        float real = fftOutput[i];
        float imag = (i == 0 || i == numBins) ? 0 : fftOutput[bufferSize - i];
        magnitudes[i] = sqrt(real * real + imag * imag);
    }

    // Compute band energies from the fresh magnitudes
    computeBandEnergies();
}

void FFTProcessor::computeBandEnergies() {
    size_t numBins = magnitudes.size();
    if (numBins == 0) return;

    // Frequency ranges (approximate for 44100 Hz sample rate, bufferSize=1024):
    // Bin width = sampleRate / bufferSize ≈ 43 Hz
    float sampleRate = 44100.0f;  // typical default
    float binWidth = sampleRate / static_cast<float>(bufferSize);

    // -- Band boundaries (in Hz) --
    // Bass:    20 Hz  – 250 Hz
    // Mid:     250 Hz – 4000 Hz
    // Treble:  4000 Hz – Nyquist (sampleRate/2)
    int bassEnd   = static_cast<int>(250.0f / binWidth);           // up to 250 Hz
    int midEnd    = static_cast<int>(4000.0f / binWidth);          // up to 4 kHz
    int trebleEnd = numBins;                                       // up to Nyquist

    // Clamp to valid range
    bassEnd   = min(bassEnd, static_cast<int>(numBins));
    midEnd    = min(midEnd, static_cast<int>(numBins));

    // Start bass from bin 1 to skip DC component (bin 0)
    int bassStart = 1;

    // Compute raw band energies (mean magnitude per band)
    float newBass = 0.0f, newMid = 0.0f, newTreble = 0.0f;

    int bassCount = bassEnd - bassStart;
    if (bassCount > 0) {
        for (int i = bassStart; i < bassEnd; ++i)
            newBass += magnitudes[i];
        newBass /= static_cast<float>(bassCount);
    }

    int midCount = midEnd - bassEnd;
    if (midCount > 0) {
        for (int i = bassEnd; i < midEnd; ++i)
            newMid += magnitudes[i];
        newMid /= static_cast<float>(midCount);
    }

    int trebleCount = trebleEnd - midEnd;
    if (trebleCount > 0) {
        for (int i = midEnd; i < trebleEnd; ++i)
            newTreble += magnitudes[i];
        newTreble /= static_cast<float>(trebleCount);
    }

    // Assign raw values
    bass = newBass;
    mid = newMid;
    treble = newTreble;

    // EMA smoothing for attenuated versions
    // bassAtt = bassAtt * (1 - SMOOTHING) + newBass * SMOOTHING
    // On first frame, initialise from raw value.
    if (bassAtt == 0.0f && bass != 0.0f) {
        bassAtt = bass;
        midAtt = mid;
        trebleAtt = treble;
    } else {
        float s = SMOOTHING;
        bassAtt   = bassAtt   * (1.0f - s) + bass   * s;
        midAtt    = midAtt    * (1.0f - s) + mid    * s;
        trebleAtt = trebleAtt * (1.0f - s) + treble * s;
    }
}

const vector<float>& FFTProcessor::getMagnitudes() const {
    return magnitudes;
}
