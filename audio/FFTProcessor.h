#ifndef FFTPROCESSOR_H
#define FFTPROCESSOR_H

#include <vector>

using namespace std;

class FFTProcessor {
public:
    FFTProcessor(size_t bufferSize);
    ~FFTProcessor();

    void computeFFT(const vector<float>& audioData);
    const vector<float>& getMagnitudes() const;

    // Frequency band energy accessors
    float getBass() const { return bass; }
    float getMid() const { return mid; }
    float getTreble() const { return treble; }
    float getBassAtt() const { return bassAtt; }
    float getMidAtt() const { return midAtt; }
    float getTrebleAtt() const { return trebleAtt; }

private:
    void computeBandEnergies();

    size_t bufferSize;
    vector<float> magnitudes;
    float* fftInput;
    float* fftOutput;
    void* fftPlan;  // Plan type depends on FFTW version

    // Frequency band loudness (raw)
    float bass;
    float mid;
    float treble;

    // Smoothed/attenuated versions (EMA)
    float bassAtt;
    float midAtt;
    float trebleAtt;

    // EMA smoothing factor (0..1, higher = slower response)
    static constexpr float SMOOTHING = 0.15f;
};

#endif // FFTPROCESSOR_H
