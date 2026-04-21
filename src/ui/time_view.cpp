#include "time_view.hpp"
#include "../PluginProcessor.h"
#include "plugin_ui.hpp"

void TimeView::UpdateGui() {
    std::ranges::copy(p_.dsp_state_.coeffs_, coeff_buffer_.begin());
    if (p_.dsp_state_.param.fir_coeff_len != 0) {
        coeff_buffer_[p_.dsp_state_.param.fir_coeff_len] = coeff_buffer_[p_.dsp_state_.param.fir_coeff_len - 1];
    }
    repaint();
}

void TimeView::paint(juce::Graphics& g) {
    g.fillAll(ui::green_bg);

    // 获取图表bound
    auto b = getLocalBounds();
    b.reduce(2, 2);
    auto bf = b.toFloat();
    g.setColour(ui::black_bg);
    g.fillRect(b);

    g.reduceClipRegion(b);
    // 绘制实际使用的，如果是mouseUp后就是实际的
    g.setColour(ui::line_fore);
    float lasty = juce::jmap(coeff_buffer_[0], -1.0f, 1.0f, bf.getBottom(), bf.getY());
    float lastx = bf.getX();
    float const fcoeff_len = static_cast<float>(p_.param_fir_coeff_len_->get());
    for (int x = 0; x < b.getWidth(); ++x) {
        size_t const idx = static_cast<size_t>(static_cast<float>(x) * fcoeff_len / static_cast<float>(b.getWidth()));
        float const val = coeff_buffer_[idx];
        float const y = juce::jmap(val, -1.0f, 1.0f, bf.getBottom(), bf.getY());
        float const xx = static_cast<float>(x) + bf.getX();
        g.drawLine(lastx, lasty, xx, y);
        lastx = xx;
        lasty = y;
    }

    if (display_waveform_) {
        // 绘制自定义波形
        g.setColour(ui::active_bg);
        lasty = juce::jmap(p_.dsp_state_.param.custom_coeffs_[0], -1.0f, 1.0f, bf.getBottom(), bf.getY());
        lastx = bf.getX();
        for (int x = 0; x < b.getWidth(); ++x) {
            size_t const idx =
                static_cast<size_t>(static_cast<float>(x) * fcoeff_len / static_cast<float>(b.getWidth()));
            float const val = p_.dsp_state_.param.custom_coeffs_[idx];
            float const y = juce::jmap(val, -1.0f, 1.0f, bf.getBottom(), bf.getY());
            float const xx = static_cast<float>(x) + bf.getX();
            g.drawLine(lastx, lasty, xx, y);
            lastx = xx;
            lasty = y;
        }
    }
}

void TimeView::mouseDrag(const juce::MouseEvent& e) {
    // 获取图表bound
    auto b = getLocalBounds();
    b.reduce(2, 2);

    auto pos = e.getPosition();
    pos.x = std::clamp(pos.x, b.getX(), b.getRight());
    pos.y = std::clamp(pos.y, b.getY(), b.getBottom());

    float const fcoeff_len = static_cast<float>(p_.param_fir_coeff_len_->get());
    auto bf = b.toFloat();
    size_t idx = static_cast<size_t>((static_cast<float>(pos.getX()) - bf.getX()) * fcoeff_len
                                     / static_cast<float>(bf.getWidth()));
    idx = std::clamp<size_t>(idx, 0, p_.param_fir_coeff_len_->get() - 1);

    float val = juce::jmap(static_cast<float>(pos.y), bf.getY(), bf.getBottom(), 1.0f, -1.0f);
    if (e.mods.isRightButtonDown()) {
        val = 0;
    }

    coeff_buffer_[idx] = val;
    p_.dsp_state_.param.custom_coeffs_[idx] = val;

    repaint();
    if (auto* parent = getParentComponent(); parent != nullptr) {
        static_cast<PluginUi*>(parent)->UpdateGuiFromTimeView();
    }
}

void TimeView::RepaintTimeAndSpectralView() {
    repaint();
    if (auto* parent = getParentComponent(); parent != nullptr) {
        static_cast<PluginUi*>(parent)->repaint();
    }
}

void TimeView::mouseUp(const juce::MouseEvent& e) {
    std::ignore = e;
    SendCoeffs();
}

void TimeView::SendCoeffs() {
    p_.dsp_state_.param.fir_source = dsp::DspParam::FirSource::kTimeCoeff;
    p_.dsp_state_.param.should_update_fir_ = true;
}

void TimeView::CopyCoeffesToCustom() {
    std::ranges::copy(p_.dsp_state_.coeffs_, p_.dsp_state_.param.custom_coeffs_.begin());
    std::ranges::copy(p_.dsp_state_.param.custom_coeffs_, coeff_buffer_.begin());
    repaint();
}

void TimeView::ClearCustomCoeffs() {
    std::ranges::fill(p_.dsp_state_.param.custom_coeffs_, float{});
    repaint();
}
