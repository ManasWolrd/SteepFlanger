#pragma once
#include "global.hpp"
#include "pluginshared/bpm_sync_ui.hpp"
#include "pluginshared/component.hpp"
#include "pluginshared/preset_panel.hpp"
#include "spectral_view.hpp"
#include "time_view.hpp"

//==============================================================================
class PluginUi final
    : public juce::Component
    , public juce::Timer {
public:
    explicit PluginUi(SteepFlangerAudioProcessor&);
    ~PluginUi() override;

    //==============================================================================
    void paint(juce::Graphics&) override;
    void resized() override;

    void UpdateGui() {
        timeview_.UpdateGui();
        spectralview_.UpdateGui();
    }

    void UpdateGuiFromTimeView() {
        spectralview_.UpdateGui();
    }

    void timerCallback() override;
    std::function<void(int, int)> on_want_new_size;
private:
    void TrySetSize(int width, int height) {
        if (on_want_new_size) {
            on_want_new_size(width, height);
        }
    }

    void SetIirMode(bool is_iir);

    SteepFlangerAudioProcessor& p_;
    pluginshared::PresetPanel preset_panel_;

    juce::Rectangle<int> lfo_bound_;
    juce::Label lfo_title_{"lfo", "lfo"};
    ui::Dial delay_{"delay"};
    ui::Dial depth_{"depth"};
    ui::BpmSyncDial speed_{"speed"};
    ui::Dial phase_{"phase"};
    ui::Dial drywet_{"drywet"};
    ui::FlatButton lfo_reset_phase_;

    juce::Rectangle<int> fir_bound_;
    ui::Switch iir_mode_{"iir", "fir"};
    ui::Dial cutoff_{"cutoff"};
    ui::Dial coeff_len_{"steep"};
    ui::Dial side_lobe_{"side_lobe"};
    ui::Switch minum_phase_{"minum_phase"};
    ui::Switch highpass_{"highpass"};
    ui::Switch custom_{"custom"};

    juce::Rectangle<int> feedback_bound_;
    juce::Label feedback_title_{"feedback", "feedback"};
    ui::Dial fb_value_{"feedback"};
    ui::Dial fb_damp_{"damp"};
    ui::FlatButton panic_;

    juce::Rectangle<int> barber_bound_;
    juce::Label barber_title_{"barberpole", "barberpole"};
    ui::Switch barber_enable_{"enable"};
    ui::Dial barber_phase_{"phase"};
    ui::BpmSyncDial barber_speed_{"speed"};
    ui::Dial barber_stereo_{"stereo"};
    ui::FlatButton barber_reset_phase_;

    TimeView timeview_;
    SpectralView spectralview_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginUi)
};
