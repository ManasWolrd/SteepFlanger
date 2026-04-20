#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include "pluginshared/component.hpp"
#include "qwqdsp/spectral/real_fft.hpp"

class TimeView;

class SpectralView : public juce::Component {
public:
    SpectralView(TimeView& time)
        : time_(time) {
        addAndMakeVisible(title_);
        fft_.Init(kGainFFTSize);
    }

    void paint(juce::Graphics& g) override;

    void resized() override {
        auto b = getLocalBounds();
        title_.setBounds(b.removeFromTop(static_cast<int>(title_.getFont().getHeight())));
    }

    void UpdateGui();

    void mouseDown(const juce::MouseEvent& e) override {
        mouseDrag(e);
    }

    void mouseDrag(const juce::MouseEvent& e) override;

    void mouseUp(const juce::MouseEvent& e) override;
private:
    static constexpr size_t kGainFFTSize = 1024;
    static constexpr size_t kGainNumBins = qwqdsp_spectral::RealFFT::NumBins(kGainFFTSize);

    TimeView& time_;
    juce::Label title_{"", "Responce"};
    std::array<float, kGainNumBins> gains_{};
    qwqdsp_spectral::RealFFT fft_;
};
