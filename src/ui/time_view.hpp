#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include "pluginshared/component.hpp"
#include "global.hpp"

class SteepFlangerAudioProcessor;

class TimeView : public juce::Component {
public:
    TimeView(SteepFlangerAudioProcessor& p)
        : p_(p) {
        addAndMakeVisible(title_);
        reload_.onClick = [this] { SendCoeffs(); };
        reload_.setButtonText("reload");
        addAndMakeVisible(reload_);

        copy_.onClick = [this] { CopyCoeffesToCustom(); };
        copy_.setButtonText("copy");
        addAndMakeVisible(copy_);

        clear_.onClick = [this] { ClearCustomCoeffs(); };
        clear_.setButtonText("clear");
        addAndMakeVisible(clear_);

        display_custom_.setToggleState(true, juce::dontSendNotification);
        addAndMakeVisible(display_custom_);
        display_custom_.onStateChange = [this] { RepaintTimeAndSpectralView(); };
    }

    void paint(juce::Graphics& g) override;

    void mouseDown(const juce::MouseEvent& e) override {
        mouseDrag(e);
    }

    void mouseDrag(const juce::MouseEvent& e) override;

    void mouseUp(const juce::MouseEvent& e) override;

    void SendCoeffs();

    void CopyCoeffesToCustom();

    void ClearCustomCoeffs();

    void resized() override {
        auto b = getLocalBounds();
        auto top = b.removeFromTop(static_cast<int>(title_.getFont().getHeight() * 1.5f));
        reload_.setBounds(top.removeFromRight(60).reduced(1, 1));
        copy_.setBounds(top.removeFromRight(50).reduced(1, 1));
        clear_.setBounds(top.removeFromRight(50).reduced(1, 1));
        display_custom_.setBounds(top.removeFromRight(70).reduced(1, 1));
        title_.setBounds(top);
    }

    void UpdateGui();
private:
    void RepaintTimeAndSpectralView();

    SteepFlangerAudioProcessor& p_;
    juce::Label title_{"", "Time view"};
    ui::FlatButton reload_;
    ui::FlatButton copy_;
    ui::FlatButton clear_;
    ui::Switch display_custom_{"show ctm"};
    std::array<float, global::kMaxCoeffLen + 1> coeff_buffer_{};

    friend class SpectralView;
};
