#include "plugin_ui.hpp"

#include "../PluginProcessor.h"

#include "qwqdsp/convert.hpp"
#include "qwqdsp/oscillator/mcf_sine_osc.hpp"

PluginUi::PluginUi(SteepFlangerAudioProcessor& p)
    : p_(p)
    , preset_panel_(*p.preset_manager_)
    , timeview_(p)
    , spectralview_(timeview_) {
    auto& apvts = *p.value_tree_;

    addAndMakeVisible(preset_panel_);

    addAndMakeVisible(lfo_title_);
    delay_.BindParam(apvts, "delay");
    addAndMakeVisible(delay_);
    depth_.BindParam(apvts, "depth");
    addAndMakeVisible(depth_);
    speed_.BindParam(p.delay_lfo_state_);
    addAndMakeVisible(speed_);
    phase_.BindParam(apvts, "phase");
    addAndMakeVisible(phase_);
    lfo_reset_phase_.setButtonText("reset phase");
    lfo_reset_phase_.onClick = [this] {
        // juce::ScopedLock _{p_.getCallbackLock()};
        // p_.dsp_.SetLFOPhase(0);
    };
    addAndMakeVisible(lfo_reset_phase_);
    drywet_.BindParam(p.param_drywet_);
    addAndMakeVisible(drywet_);

    cutoff_.BindParam(apvts, "cutoff");
    addAndMakeVisible(cutoff_);
    coeff_len_.BindParam(apvts, "coeff_len");
    addAndMakeVisible(coeff_len_);
    side_lobe_.BindParam(apvts, "side_lobe");
    addAndMakeVisible(side_lobe_);
    minum_phase_.BindParam(apvts, "minum_phase");
    addAndMakeVisible(minum_phase_);
    iir_mode_.onClick = [this] { SetIirMode(iir_mode_.getToggleState()); };
    iir_mode_.BindParam(p.param_iir_mode_);
    addAndMakeVisible(iir_mode_);
    highpass_.BindParam(apvts, "highpass");
    addAndMakeVisible(highpass_);
    custom_.onClick = [this] {
        if (custom_.getToggleState()) {
            TrySetSize(600, 264 + 200 + 30);
            timeview_.setVisible(true);
            spectralview_.setVisible(true);
        }
        else {
            TrySetSize(600, 264 + 30);
            timeview_.setVisible(false);
            spectralview_.setVisible(false);
        }
    };
    addAndMakeVisible(custom_);

    fb_value_.BindParam(apvts, "fb_value");
    addAndMakeVisible(fb_value_);
    panic_.setButtonText("panic");
    panic_.onClick = [&p] {
        // p.dsp_.Reset();
    };
    addAndMakeVisible(panic_);
    fb_damp_.BindParam(apvts, "fb_damp");
    addAndMakeVisible(fb_damp_);
    addAndMakeVisible(feedback_title_);

    addAndMakeVisible(barber_title_);
    barber_phase_.BindParam(apvts, "barber_phase");
    addAndMakeVisible(barber_phase_);
    barber_speed_.BindParam(p.barber_lfo_state_);
    addAndMakeVisible(barber_speed_);
    barber_enable_.BindParam(apvts, "barber_enable");
    addAndMakeVisible(barber_enable_);
    barber_reset_phase_.setButtonText("reset phase");
    barber_reset_phase_.onClick = [this] {
        // juce::ScopedLock _{p_.getCallbackLock()};
        // p_.dsp_.SetBarberLFOPhase(0);
    };
    addAndMakeVisible(barber_reset_phase_);
    barber_stereo_.BindParam(p.param_barber_stereo_);
    addAndMakeVisible(barber_stereo_);

    addAndMakeVisible(timeview_);
    addAndMakeVisible(spectralview_);

    setSize(600, 264 + 30);
    custom_.setToggleState(p.dsp_state_.param.fir_source != dsp::DspParam::FirSource::kWindowSinc, juce::sendNotificationSync);
    iir_mode_.onClick();
    startTimerHz(30);
}

PluginUi::~PluginUi() {}

//==============================================================================
void PluginUi::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour{22, 27, 32});

    auto b = getLocalBounds();
    g.setColour(ui::green_bg);
    g.fillRect(b.removeFromTop(30));
    b.removeFromTop(2);
    g.fillRect(lfo_bound_);
    g.fillRect(fir_bound_);
    g.fillRect(feedback_bound_);
    g.fillRect(barber_bound_);
}

void PluginUi::resized() {
    auto b = getLocalBounds();
    preset_panel_.setBounds(b.removeFromTop(30));
    b.removeFromTop(2);
    {
        auto topblock = b.removeFromTop(125);
        {
            auto lfo_block = topblock.removeFromLeft(80 * 4);
            lfo_bound_ = lfo_block;
            auto lfo_block_top = lfo_block.removeFromTop(25);
            lfo_reset_phase_.setBounds(lfo_block_top.removeFromRight(100).reduced(1, 1));
            lfo_title_.setBounds(lfo_block_top);
            delay_.setBounds(lfo_block.removeFromLeft(64));
            ;
            depth_.setBounds(lfo_block.removeFromLeft(64));
            speed_.setBounds(lfo_block.removeFromLeft(64));
            phase_.setBounds(lfo_block.removeFromLeft(64));
            drywet_.setBounds(lfo_block.removeFromLeft(64));
        }
        topblock.removeFromLeft(8);
        {
            auto fir_block = topblock.removeFromLeft(80 * 4);
            fir_bound_ = fir_block;
            {
                auto fir_title = fir_block.removeFromTop(25).reduced(2, 0);
                minum_phase_.setBounds(fir_title.removeFromRight(100).reduced(2, 0));
                highpass_.setBounds(fir_title.removeFromRight(70).reduced(2, 0));
                custom_.setBounds(fir_title.removeFromRight(60).reduced(2, 0));
                iir_mode_.setBounds(fir_title);
            }
            cutoff_.setBounds(fir_block.removeFromLeft(80));
            ;
            coeff_len_.setBounds(fir_block.removeFromLeft(80));
            ;
            side_lobe_.setBounds(fir_block.removeFromLeft(80));
            ;
        }
    }
    b.removeFromTop(8);
    {
        auto bottom_block = b.removeFromTop(125);
        {
            auto feedback_block = bottom_block.removeFromLeft(80 * 3);
            feedback_bound_ = feedback_block;
            feedback_title_.setBounds(feedback_block.removeFromTop(25));
            {
                auto button_block = feedback_block.removeFromLeft(80);
                panic_.setBounds(button_block.withSizeKeepingCentre(button_block.getWidth(), 30));
            }
            fb_value_.setBounds(feedback_block.removeFromLeft(80));
            fb_damp_.setBounds(feedback_block.removeFromLeft(80));
        }
        bottom_block.removeFromLeft(8);
        {
            auto barber_block = bottom_block.removeFromLeft(250);
            barber_bound_ = barber_block;
            auto barber_title_bound = barber_block.removeFromTop(25);
            barber_reset_phase_.setBounds(barber_title_bound.removeFromRight(100).reduced(2));
            barber_enable_.setBounds(barber_title_bound.removeFromRight(60).reduced(2));
            barber_title_.setBounds(barber_title_bound);
            barber_phase_.setBounds(barber_block.removeFromLeft(80));
            barber_speed_.setBounds(barber_block.removeFromLeft(80));
            barber_stereo_.setBounds(barber_block);
        }
    }
    b.removeFromTop(8);
    if (custom_.getToggleState()) {
        auto graphic_block = b.removeFromTop(200);
        auto time_block = graphic_block.removeFromLeft(graphic_block.getWidth() / 2);
        time_block.removeFromRight(4);
        timeview_.setBounds(time_block);
        auto spectral_block = graphic_block;
        spectral_block.removeFromRight(4);
        spectralview_.setBounds(spectral_block);
    }
}

void PluginUi::timerCallback() {
    if (p_.dsp_state_.have_new_coeff_.exchange(false)) {
        UpdateGui();
    }
}

void PluginUi::SetIirMode(bool is_iir) {
    minum_phase_.setVisible(!is_iir);
    custom_.setVisible(!is_iir);
    fb_damp_.setEnabled(!is_iir);
    fb_value_.setEnabled(!is_iir);

    if (is_iir) {
        coeff_len_.BindParam(p_.param_iir_filter_num_);
        side_lobe_.BindParam(p_.param_iir_ripple_);
    }
    else {
        coeff_len_.BindParam(p_.param_fir_coeff_len_);
        side_lobe_.BindParam(p_.param_fir_side_lobe_);
    }

    coeff_len_.label.setText(is_iir ? "N filter" : "coeff len", juce::dontSendNotification);
    side_lobe_.label.setText(is_iir ? "ripple" : "side lobe", juce::dontSendNotification);
}
