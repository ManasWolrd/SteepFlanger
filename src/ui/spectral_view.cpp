#include "spectral_view.hpp"
#include "qwqdsp/oscillator/mcf_sine_osc.hpp"
#include "time_view.hpp"
#include "../PluginProcessor.h"
#include "qwqdsp/convert.hpp"

void SpectralView::paint(juce::Graphics& g) {
    g.fillAll(ui::green_bg);

    // 获取图表bound
    auto b = getLocalBounds();
    b.removeFromTop(title_.getHeight());
    b.reduce(2, 8);
    auto text_bound = b.removeFromLeft(36).toFloat();
    auto bf = b.toFloat();
    g.setColour(ui::black_bg);
    g.fillRect(b);
    
    // 绘制频谱音量数字
    float const fcoeff_len = static_cast<float>(time_.p_.dsp_state_.param.fir_coeff_len);
    constexpr size_t kNumLines = 6;
    float const centerx = text_bound.getCentreX();
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font{juce::FontOptions{}.withHeight(12)});
    for (size_t i = 0; i < kNumLines; ++i) {
        float const centery = text_bound.getY() + static_cast<float>(i) * static_cast<float>(text_bound.getHeight()) / (kNumLines - 1.0f);
        juce::Rectangle<float> text{0.0, 0.0, text_bound.getWidth(), 12.0f};
        text = text.withCentre({centerx, centery});
        int const val = -static_cast<int>(i) * 20;
        g.drawText(juce::String{val}, text, juce::Justification::right);
    }

    // 绘制超采样频谱
    g.setColour(ui::line_fore);
    float lasty = juce::jmap(gains_[0], bf.getBottom(), bf.getY());
    float lastx = bf.getX();
    for (int x = 0; x < b.getWidth(); ++x) {
        size_t idx = static_cast<size_t>(static_cast<float>(static_cast<size_t>(x) * gains_.size()) / static_cast<float>(b.getWidth()));
        idx = std::min(idx, gains_.size() - 1);
        float const val = gains_[idx];
        float const y = juce::jmap(val, bf.getBottom(), bf.getY());
        float const xx = static_cast<float>(x) + bf.getX();
        g.drawLine(lastx, lasty, xx, y);
        lastx = xx;
        lasty = y;
    }

    // 绘制自定义频谱
    if (time_.display_custom_.getToggleState()) {
        g.setColour(ui::active_bg);
        lasty = juce::jmap(time_.p_.dsp_state_.param.custom_spectral_gains[0], bf.getBottom(), bf.getY());
        lastx = bf.getX();
        for (int x = 0; x < b.getWidth(); ++x) {
            size_t const idx = static_cast<size_t>(static_cast<float>(static_cast<float>(x) * fcoeff_len) / static_cast<float>(b.getWidth()));
            float const val = time_.p_.dsp_state_.param.custom_spectral_gains[idx];
            float const y = juce::jmap(val, bf.getBottom(), bf.getY());
            float const xx = static_cast<float>(x) + bf.getX();
            g.drawLine(lastx, lasty, xx, y);
            lastx = xx;
            lasty = y;
        }
    }
}

void SpectralView::UpdateGui() {
    std::array<float, kGainFFTSize> fft_buffer{};
    std::copy_n(time_.coeff_buffer_.begin(), time_.p_.dsp_state_.param.fir_coeff_len, fft_buffer.begin());
    fft_.FFTGainPhase(fft_buffer, gains_);

    for (auto& x : gains_) {
        x = qwqdsp::convert::Gain2Db<-100.0f>(x);
    }

    for (auto& x : gains_) {
        x = std::clamp((x + 100.0f) / 100.0f, 0.0f, 1.0f);
    }

    repaint();
}

void SpectralView::mouseDrag(const juce::MouseEvent& e) {
    // 获取图表bound
    auto b = getLocalBounds();
    b.removeFromTop(title_.getHeight());
    b.reduce(2, 8);
    b.removeFromLeft(36).toFloat();
    auto bf = b.toFloat();

    auto pos = e.getPosition();
    pos.x = std::clamp(pos.x, b.getX(), b.getRight());
    pos.y = std::clamp(pos.y, b.getY(), b.getBottom());

    size_t const coeff_len = time_.p_.dsp_state_.param.fir_coeff_len;
    float const fcoeff_len = static_cast<float>(coeff_len);
    size_t idx = static_cast<size_t>((static_cast<float>(pos.getX()) - bf.getX()) * fcoeff_len / bf.getWidth());
    idx = std::clamp<size_t>(idx, 0, coeff_len - 1);

    float val = juce::jmap(static_cast<float>(pos.y), bf.getY(), bf.getBottom(), 1.0f, 0.0f);
    if (e.mods.isRightButtonDown()) {
        val = 0;
    }
    
    time_.p_.dsp_state_.param.custom_spectral_gains[idx] = val;

    // 加法合成
    std::array<qwqdsp_oscillator::MCFSineOsc, global::kMaxCoeffLen> oscs;
    std::array<float, global::kMaxCoeffLen> true_gains;
    for (size_t i = 0; i < coeff_len; ++i) {
        oscs[i].Reset(static_cast<float>(i) * std::numbers::pi_v<float> / fcoeff_len, 0.0f);
        float const db = std::lerp(-101.0f, 0.0f, time_.p_.dsp_state_.param.custom_spectral_gains[i]);
        if (db < -100.0f) {
            true_gains[i] = 0;
        }
        else {
            true_gains[i] = qwqdsp::convert::Db2Gain(db);
        }
    }
    
    for (size_t tidx = 0; tidx < coeff_len; ++tidx) {
        float sum{};
        for (size_t fidx = 0; fidx < coeff_len; ++fidx) {
            sum += true_gains[fidx] * oscs[fidx].Tick();
        }
        time_.coeff_buffer_[tidx] = sum;
        time_.p_.dsp_state_.param.custom_coeffs_[tidx] = sum;
    }

    UpdateGui();
    time_.repaint();
}

void SpectralView::mouseUp(const juce::MouseEvent& e) {
    std::ignore = e;
    time_.SendCoeffs();
}
