#include "steep_flanger.hpp"

#include "qwqdsp/convert.hpp"
#include "qwqdsp/polymath.hpp"

#if defined(VEC8_2)
void SteepFlanger::ProcessVec8_2(
#else
void SteepFlanger::ProcessVec8(
#endif
    float* left_ptr, float* right_ptr, size_t len,
    SteepFlangerParameter& param
) noexcept {
    size_t cando = len;
    while (cando != 0) {
        size_t num_process = std::min<size_t>(512, cando);
        cando -= num_process;

        if (param.should_update_fir_.exchange(false)) {
            UpdateCoeff(param);
        }

        constexpr float kWarpFactor = -0.8f;
        float warp_drywet = param.drywet;
        warp_drywet = 2.0f * warp_drywet - 1.0f;
        warp_drywet = (warp_drywet - kWarpFactor) / (1.0f - warp_drywet * kWarpFactor);
        warp_drywet = 0.5f * warp_drywet + 0.5f;
        warp_drywet = std::clamp(warp_drywet, 0.0f, 1.0f);
        float feedback_mul = warp_drywet * param.feedback;

        float const damp_pitch = param.damp_pitch;
        float const damp_freq = qwqdsp::convert::Pitch2Freq(damp_pitch);
        float const damp_w = qwqdsp::convert::Freq2W(damp_freq, fs_);
        damp_lowpass_coeff_ = damp_.ComputeCoeff(damp_w);

        barber_phase_smoother_.SetTarget(param.barber_phase);
        barber_oscillator_.SetFreq(param.barber_speed, fs_);

        // update delay times
        phase_ += param.lfo_freq / fs_ * static_cast<float>(num_process);
        float right_phase = phase_ + param.lfo_phase;
        {
            float t;
            phase_ = std::modf(phase_, &t);
            right_phase = std::modf(right_phase, &t);
        }
        float left_phase = phase_;

        simd::Float128 lfo_modu;
        lfo_modu[0] = qwqdsp::polymath::SinPi(left_phase * std::numbers::pi_v<float>);
        lfo_modu[1] = qwqdsp::polymath::SinPi(right_phase * std::numbers::pi_v<float>);

        float const delay_samples = param.delay_ms * fs_ / 1000.0f;
        float const depth_samples = param.depth_ms * fs_ / 1000.0f;
        simd::Float128 target_delay_samples = delay_samples + lfo_modu * depth_samples;
        target_delay_samples = simd::Max128(target_delay_samples, simd::BroadcastF128(0.0f));
        float const delay_time_smooth_factor = 1.0f - std::exp(-1.0f / (fs_ / static_cast<float>(num_process) * kDelaySmoothMs / 1000.0f));
        last_exp_delay_samples_ += delay_time_smooth_factor * (target_delay_samples - last_exp_delay_samples_);
        auto curr_num_notch = last_delay_samples_;
        auto delta_num_notch = (last_exp_delay_samples_ - curr_num_notch) / static_cast<float>(num_process);

        float curr_damp_coeff = last_damp_lowpass_coeff_;
        float delta_damp_coeff = (damp_lowpass_coeff_ - curr_damp_coeff) / (static_cast<float>(num_process));

        float inv_samples = 1.0f / static_cast<float>(num_process);
        alignas(32) std::array<simd::Float256, kSIMDMaxCoeffLen / 8> delta_coeffs;
        auto* coeffs_ptr = (simd::Float256*)(coeffs_.data());
        auto* last_coeffs_ptr = (simd::Float256*)(last_coeffs_.data());
        size_t const coeff_len_div8 = (coeff_len_ + 7) / 8;
        float const wet_mix = param.drywet;
        simd::Float256 dry_coeff{1.0f - wet_mix, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < coeff_len_div8; ++i) {
            simd::Float256 target_wet_coeff = coeffs_ptr[i] * wet_mix + dry_coeff;
            delta_coeffs[i] = (target_wet_coeff - last_coeffs_ptr[i]) * inv_samples;
            dry_coeff = simd::Float256{};
        }

        // fir polyphase filtering
        if (!param.barber_enable) {
            for (size_t j = 0; j < num_process; ++j) {
                curr_num_notch += delta_num_notch;
                curr_damp_coeff += delta_damp_coeff;

                for (size_t i = 0; i < coeff_len_div8; ++i) {
                    last_coeffs_ptr[i] += delta_coeffs[i];
                }
    
                float left_sum = 0;
                float const left_num_notch = curr_num_notch[0];
                simd::Float256 current_delay;
                current_delay[0] = 0;
                current_delay[1] = left_num_notch;
                current_delay[2] = left_num_notch * 2;
                current_delay[3] = left_num_notch * 3;
                current_delay[4] = left_num_notch * 4;
                current_delay[5] = left_num_notch * 5;
                current_delay[6] = left_num_notch * 6;
                current_delay[7] = left_num_notch * 7;
                auto delay_inc = simd::BroadcastF256(left_num_notch * 8);
                delay_left_.Push(*left_ptr + left_fb_ * feedback_mul);
                for (size_t i = 0; i < coeff_len_div8; ++i) {
                    auto taps_out = delay_left_.GetAfterPush(current_delay);
                    current_delay += delay_inc;

                    taps_out *= last_coeffs_ptr[i];
                    left_sum += simd::ReduceAdd(taps_out);
                }

                float right_sum = 0;
                float const right_num_notch = curr_num_notch[1];
                current_delay[0] = 0;
                current_delay[1] = right_num_notch;
                current_delay[2] = right_num_notch * 2;
                current_delay[3] = right_num_notch * 3;
                current_delay[4] = right_num_notch * 4;
                current_delay[5] = right_num_notch * 5;
                current_delay[6] = right_num_notch * 6;
                current_delay[7] = right_num_notch * 7;
                delay_inc = simd::BroadcastF256(right_num_notch * 8);
                delay_right_.Push(*right_ptr + right_fb_ * feedback_mul);
                for (size_t i = 0; i < coeff_len_div8; ++i) {
                    auto taps_out = delay_right_.GetAfterPush(current_delay);
                    current_delay += delay_inc;
                    taps_out *= last_coeffs_ptr[i];
                    right_sum += simd::ReduceAdd(taps_out);
                }

                simd::Float128 damp_x;
                damp_x[0] = left_sum;
                damp_x[1] = right_sum;
                *left_ptr = left_sum * fir_gain_;
                *right_ptr = right_sum * fir_gain_;
                ++left_ptr;
                ++right_ptr;
                damp_x = damp_.TickLowpass(damp_x, simd::BroadcastF128(curr_damp_coeff));
                auto dc_remove = dc_.TickHighpass(damp_x, simd::BroadcastF128(0.0005f));
                left_fb_ = qwqdsp::polymath::ArctanPade(dc_remove[0]);
                right_fb_ = qwqdsp::polymath::ArctanPade(dc_remove[1]);
            }
        }
        else {
            for (size_t j = 0; j < num_process; ++j) {
                curr_damp_coeff += delta_damp_coeff;
                curr_num_notch += delta_num_notch;

                for (size_t i = 0; i < coeff_len_div8; ++i) {
                    last_coeffs_ptr[i] += delta_coeffs[i];
                }

                delay_left_.Push(*left_ptr + left_fb_ * feedback_mul);
                delay_right_.Push(*right_ptr + right_fb_ * feedback_mul);

                float const left_num_notch = curr_num_notch[0];
                float const right_num_notch = curr_num_notch[1];
                simd::Float256 left_current_delay;
                simd::Float256 right_current_delay;
                left_current_delay[0] = 0;
                left_current_delay[1] = left_num_notch;
                left_current_delay[2] = left_num_notch * 2;
                left_current_delay[3] = left_num_notch * 3;
                left_current_delay[4] = left_num_notch * 4;
                left_current_delay[5] = left_num_notch * 5;
                left_current_delay[6] = left_num_notch * 6;
                left_current_delay[7] = left_num_notch * 7;
                right_current_delay[0] = 0;
                right_current_delay[1] = right_num_notch;
                right_current_delay[2] = right_num_notch * 2;
                right_current_delay[3] = right_num_notch * 3;
                right_current_delay[4] = right_num_notch * 4;
                right_current_delay[5] = right_num_notch * 5;
                right_current_delay[6] = right_num_notch * 6;
                right_current_delay[7] = right_num_notch * 7;
                auto left_delay_inc = simd::BroadcastF256(left_num_notch * 8);
                auto right_delay_inc = simd::BroadcastF256(right_num_notch * 8);

                auto const addition_rotation = std::polar(1.0f, barber_phase_smoother_.Tick() * std::numbers::pi_v<float> * 2);
                barber_oscillator_.Tick();
                auto const rotation_once = barber_oscillator_.GetCpx() * addition_rotation;
                auto const rotation_2 = rotation_once * rotation_once;
                auto const rotation_3 = rotation_once * rotation_2;
                auto const rotation_4 = rotation_2 * rotation_2;
                auto const rotation_5 = rotation_2 * rotation_3;
                auto const rotation_6 = rotation_3 * rotation_3;
                auto const rotation_7 = rotation_3 * rotation_4;
                auto const rotation_8 = rotation_4 * rotation_4;
                Complex32x8 left_rotation_coeff;
                left_rotation_coeff.re[0] = 1;
                left_rotation_coeff.re[1] = rotation_once.real();
                left_rotation_coeff.re[2] = rotation_2.real();
                left_rotation_coeff.re[3] = rotation_3.real();
                left_rotation_coeff.re[4] = rotation_4.real();
                left_rotation_coeff.re[5] = rotation_5.real();
                left_rotation_coeff.re[6] = rotation_6.real();
                left_rotation_coeff.re[7] = rotation_7.real();
                left_rotation_coeff.im[0] = 0;
                left_rotation_coeff.im[1] = rotation_once.imag();
                left_rotation_coeff.im[2] = rotation_2.imag();
                left_rotation_coeff.im[3] = rotation_3.imag();
                left_rotation_coeff.im[4] = rotation_4.imag();
                left_rotation_coeff.im[5] = rotation_5.imag();
                left_rotation_coeff.im[6] = rotation_6.imag();
                left_rotation_coeff.im[7] = rotation_7.imag();
                Complex32x8 right_rotation_coeff = left_rotation_coeff;
                right_rotation_coeff *= Complex32x8{
                    .re = simd::BroadcastF256(std::cos(param.barber_stereo_phase)),
                    .im = simd::BroadcastF256(std::sin(param.barber_stereo_phase))
                };
                Complex32x8 left_rotation_mul{
                    .re = simd::BroadcastF256(rotation_8.real()),
                    .im = simd::BroadcastF256(rotation_8.imag())
                };
                Complex32x8 right_rotation_mul = left_rotation_mul;

                float left_re_sum = 0;
                float left_im_sum = 0;
                float right_re_sum = 0;
                float right_im_sum = 0;
                for (size_t i = 0; i < coeff_len_div8; ++i) {
                    auto left_taps_out = delay_left_.GetAfterPush(left_current_delay);
                    auto right_taps_out = delay_right_.GetAfterPush(right_current_delay);
                    left_current_delay += left_delay_inc;
                    right_current_delay += right_delay_inc;

                    left_taps_out *= last_coeffs_ptr[i];
                    auto temp = left_taps_out * left_rotation_coeff.re;
                    left_re_sum += simd::ReduceAdd(temp);
                    temp = left_taps_out * left_rotation_coeff.im;
                    left_im_sum += simd::ReduceAdd(temp);

                    right_taps_out *= last_coeffs_ptr[i];
                    temp = right_taps_out * right_rotation_coeff.re;
                    right_re_sum += simd::ReduceAdd(temp);
                    temp = right_taps_out * right_rotation_coeff.im;
                    right_im_sum += simd::ReduceAdd(temp);

                    left_rotation_coeff *= left_rotation_mul;
                    right_rotation_coeff *= right_rotation_mul;
                }
                
                auto remove_positive_spectrum = hilbert_complex_.Tick(simd::Float128{
                    left_re_sum, left_im_sum, right_re_sum, right_im_sum
                });
                // this will mirror the positive spectrum to negative domain, forming a real value signal
                auto damp_x = simd::Shuffle<simd::Float128, 0, 2, 1, 3>(remove_positive_spectrum, remove_positive_spectrum);
                *left_ptr = damp_x[0] * fir_gain_;
                *right_ptr = damp_x[1] * fir_gain_;
                ++left_ptr;
                ++right_ptr;
                damp_x = damp_.TickLowpass(damp_x, simd::BroadcastF128(curr_damp_coeff));
                auto dc_remove = dc_.TickHighpass(damp_x, simd::BroadcastF128(0.0005f));
                left_fb_ = qwqdsp::polymath::ArctanPade(dc_remove[0]);
                right_fb_ = qwqdsp::polymath::ArctanPade(dc_remove[1]);
            }

            barber_osc_keep_amp_counter_ += len;
            [[unlikely]]
            if (barber_osc_keep_amp_counter_ > barber_osc_keep_amp_need_) {
                barber_osc_keep_amp_counter_ = 0;
                barber_oscillator_.KeepAmp();
            }
        }
        last_delay_samples_ = last_exp_delay_samples_;
        last_damp_lowpass_coeff_ = damp_lowpass_coeff_;
    }
}

#if defined(VEC8_2)
void SteepFlanger::ProcessVec8_2_Iir(
#else
void SteepFlanger::ProcessVec8_Iir(
#endif
    float* left_ptr, float* right_ptr, size_t len,
    SteepFlangerParameter& param
) noexcept {
    if (param.should_update_iir_.exchange(false)) {
        int N = static_cast<int>(param.iir_num_filters);
        double eps = std::sqrt(std::pow(10.0, param.ripple / 10.0) - 1.0);
        double A = 1.0 / static_cast<double>(2 * N) * std::asinh(1.0 / eps);
        double k_re = std::sinh(A);
        double k_im = std::cosh(A);

        double cutoff = param.fir_cutoff;
        if (param.fir_highpass) {
            cutoff = std::numbers::pi_v<double> - cutoff;
        }
        cutoff = std::tan(cutoff / 2.0);

        if (last_iir_highpass_ != param.fir_highpass) {
            last_iir_highpass_ = param.fir_highpass;
            for (size_t i = 0; i < global::kIirMaxNumFilters / 8; ++i) {
                iir_filters_.iir8[i].Reset();
            }
        }

        // s域
        double k = 1.0;
        std::complex<double> half_spoles[global::kIirMaxNumFilters];
        if (!param.fir_highpass) {
            for (int i = 0; i < N; ++i) {
                double phi = (2.0 * static_cast<double>(i + 1) - 1.0) * std::numbers::pi_v<double> / static_cast<double>(4 * N);
                half_spoles[i] = cutoff * std::complex{k_re * -std::sin(phi), k_im * std::cos(phi)};
                k *= std::norm(half_spoles[i]);
            }
        }
        else {
            for (int i = 0; i < N; ++i) {
                double phi = (2.0 * static_cast<double>(i + 1) - 1.0) * std::numbers::pi_v<double> / static_cast<double>(4 * N);
                auto pole = std::complex{k_re * -std::sin(phi), k_im * std::cos(phi)};
                half_spoles[i] = cutoff / pole;
            }
        }

        // 双线性变换
        std::complex<double> half_zpoles[global::kIirMaxNumFilters];
        for (int i = 0; i < N; ++i) {
            half_zpoles[i] = (1.0 + half_spoles[i]) / (1.0 - half_spoles[i]);
                k /= std::real((1.0 - half_spoles[i]) * (1.0 - std::conj(half_spoles[i])));
        }

        // 部分分式分解
        std::complex<double> residual[global::kIirMaxNumFilters];
        for (int i = 0; i < N; ++i) {
            auto zpole = half_zpoles[i];

            std::complex<double> up = 1.0;
            std::complex<double> tmp_up = zpole + 1.0;
            if (param.fir_highpass) {
                tmp_up = zpole - 1.0;
            }
            for (int j = 0; j < N; ++j) {
                up *= tmp_up;
                up *= tmp_up;
            }

            std::complex<double> down = 1.0;
            for (int j = 0; j < N; ++j) {
                if (i == j) {
                    down *= (zpole - std::conj(half_zpoles[j]));
                }
                else {
                    down *= (zpole - half_zpoles[j]);
                    down *= (zpole - std::conj(half_zpoles[j]));
                }
            }

            residual[i] = up / down;
        }

        // 设定滤波器系数
        auto& filters = iir_filters_.iir8;
        auto* residual_ptr = &residual[0];
        auto* pole_ptr = &half_zpoles[0];
        int full_8_num = N / 8;
        int residual_8_num = N % 8;

        for (int i = 0; i < full_8_num; ++i) {
            simd::Float256 r_re{
                static_cast<float>(2 * k * residual_ptr[0].real()),
                static_cast<float>(2 * k * residual_ptr[1].real()),
                static_cast<float>(2 * k * residual_ptr[2].real()),
                static_cast<float>(2 * k * residual_ptr[3].real()),
                static_cast<float>(2 * k * residual_ptr[4].real()),
                static_cast<float>(2 * k * residual_ptr[5].real()),
                static_cast<float>(2 * k * residual_ptr[6].real()),
                static_cast<float>(2 * k * residual_ptr[7].real()),
            };
            simd::Float256 r_im{
                static_cast<float>(2 * k * residual_ptr[0].imag()),
                static_cast<float>(2 * k * residual_ptr[1].imag()),
                static_cast<float>(2 * k * residual_ptr[2].imag()),
                static_cast<float>(2 * k * residual_ptr[3].imag()),
                static_cast<float>(2 * k * residual_ptr[4].imag()),
                static_cast<float>(2 * k * residual_ptr[5].imag()),
                static_cast<float>(2 * k * residual_ptr[6].imag()),
                static_cast<float>(2 * k * residual_ptr[7].imag()),
            };
            simd::Float256 p_re{
                static_cast<float>(std::real(pole_ptr[0])),
                static_cast<float>(std::real(pole_ptr[1])),
                static_cast<float>(std::real(pole_ptr[2])),
                static_cast<float>(std::real(pole_ptr[3])),
                static_cast<float>(std::real(pole_ptr[4])),
                static_cast<float>(std::real(pole_ptr[5])),
                static_cast<float>(std::real(pole_ptr[6])),
                static_cast<float>(std::real(pole_ptr[7])),
            };
            simd::Float256 p_im{
                static_cast<float>(std::imag(pole_ptr[0])),
                static_cast<float>(std::imag(pole_ptr[1])),
                static_cast<float>(std::imag(pole_ptr[2])),
                static_cast<float>(std::imag(pole_ptr[3])),
                static_cast<float>(std::imag(pole_ptr[4])),
                static_cast<float>(std::imag(pole_ptr[5])),
                static_cast<float>(std::imag(pole_ptr[6])),
                static_cast<float>(std::imag(pole_ptr[7])),
            };
            residual_ptr += 8;
            pole_ptr += 8;
            
            filters[i].Set(SimdComplex<simd::Float256>{r_re, r_im}, SimdComplex<simd::Float256>{p_re, p_im});
        }

        if (residual_8_num != 0) {
            simd::Float256 r_re{};
            simd::Float256 r_im{};
            simd::Float256 p_re{};
            simd::Float256 p_im{};
            for (int i = 0; i < residual_8_num; ++i) {
                r_re[i] = static_cast<float>(2 * k * residual_ptr[i].real());
                r_im[i] = static_cast<float>(2 * k * residual_ptr[i].imag());
                p_re[i] = static_cast<float>(std::real(pole_ptr[i]));
                p_im[i] = static_cast<float>(std::imag(pole_ptr[i]));
            }
            filters[full_8_num].Set(SimdComplex<simd::Float256>{r_re, r_im}, SimdComplex<simd::Float256>{p_re, p_im});
        }

        iir_fir_k_ = static_cast<float>(k);
    }

    // -------------------- 处理中 --------------------
    size_t cando = len;
    while (cando != 0) {
        size_t num_process = std::min<size_t>(512, cando);
        cando -= num_process;

        float dry_mix = 1.0f - param.drywet;
        float wet_mix = param.drywet;

        float const damp_pitch = param.damp_pitch;
        float const damp_freq = qwqdsp::convert::Pitch2Freq(damp_pitch);
        float const damp_w = qwqdsp::convert::Freq2W(damp_freq, fs_);
        damp_lowpass_coeff_ = damp_.ComputeCoeff(damp_w);

        barber_phase_smoother_.SetTarget(param.barber_phase);
        barber_oscillator_.SetFreq(param.barber_speed, fs_);

        // update delay times
        phase_ += param.lfo_freq / fs_ * static_cast<float>(num_process);
        float right_phase = phase_ + param.lfo_phase;
        {
            float t;
            phase_ = std::modf(phase_, &t);
            right_phase = std::modf(right_phase, &t);
        }
        float left_phase = phase_;

        simd::Float128 lfo_modu;
        lfo_modu[0] = qwqdsp::polymath::SinPi(left_phase * std::numbers::pi_v<float>);
        lfo_modu[1] = qwqdsp::polymath::SinPi(right_phase * std::numbers::pi_v<float>);

        float const delay_samples = param.delay_ms * fs_ / 1000.0f;
        float const depth_samples = param.depth_ms * fs_ / 1000.0f;
        auto target_delay_samples = delay_samples + lfo_modu * depth_samples;
        target_delay_samples = simd::Max128(target_delay_samples, simd::BroadcastF128(1.0f));
        float const delay_time_smooth_factor = 1.0f - std::exp(-1.0f / (fs_ / static_cast<float>(num_process) * kDelaySmoothMs / 1000.0f));
        last_exp_delay_samples_ += delay_time_smooth_factor * (target_delay_samples - last_exp_delay_samples_);
        auto curr_num_notch = last_delay_samples_;
        auto delta_num_notch = (last_exp_delay_samples_ - curr_num_notch) / static_cast<float>(num_process);

        float curr_damp_coeff = last_damp_lowpass_coeff_;
        float delta_damp_coeff = (damp_lowpass_coeff_ - curr_damp_coeff) / (static_cast<float>(num_process));

        float const inv_samples = 1.0f / static_cast<float>(num_process);
        size_t num_simd_filter = (param.iir_num_filters + 7) / 8;

        if (!param.barber_enable) {
            for (size_t j = 0; j < num_process; ++j) {
                curr_num_notch += delta_num_notch;
                curr_damp_coeff += delta_damp_coeff;
    
                float const left_in = *left_ptr;
                float const right_in = *right_ptr;
                float const left_num_notch = curr_num_notch[0];
                float const right_num_notch = curr_num_notch[1];
                auto& filters = iir_filters_.iir8;

                float right_sum = left_in * iir_fir_k_;
                float left_sum = right_in * iir_fir_k_;
                float x_left = iir_x_delay_.GetBeforePush<0>(left_num_notch);
                float x_right = iir_x_delay_.GetBeforePush<1>(right_num_notch);
                for (size_t i = 0; i < num_simd_filter; ++i) {
                    auto[l, r] = filters[i].Tick(x_left, x_right, left_num_notch, left_num_notch);
                    left_sum += l;
                    right_sum += r;
                }
                iir_x_delay_.Push(left_in, right_in);

                *left_ptr = left_sum * wet_mix + left_in * dry_mix;
                *right_ptr = right_sum * wet_mix + right_in * dry_mix;
                ++left_ptr;
                ++right_ptr;
            }
        }
        else {
            for (size_t j = 0; j < num_process; ++j) {
                curr_num_notch += delta_num_notch;
                curr_damp_coeff += delta_damp_coeff;
    
                float const left_in = *left_ptr;
                float const right_in = *right_ptr;
                float const left_num_notch = curr_num_notch[0];
                float const right_num_notch = curr_num_notch[1];
                auto& filters = iir_filters_.iir8;

                std::complex<float> right_sum = 0;
                std::complex<float> left_sum = 0;
                float x_left = iir_x_delay_.GetBeforePush<0>(left_num_notch);
                float x_right = iir_x_delay_.GetBeforePush<1>(right_num_notch);

                auto const addition_rotation = std::polar(1.0f, barber_phase_smoother_.Tick() * std::numbers::pi_v<float> * 2);
                barber_oscillator_.Tick();
                auto const rotation_once = barber_oscillator_.GetCpx() * addition_rotation;
                auto const right_channel_rotation = std::polar(1.0f, param.barber_stereo_phase);
                auto left_rotate = rotation_once;
                auto right_rotate = rotation_once * right_channel_rotation;

                for (size_t i = 0; i < num_simd_filter; ++i) {
                    auto[l, r] = filters[i].TickCpx(x_left, x_right, left_num_notch, left_num_notch, left_rotate, right_rotate);
                    left_sum += l;
                    right_sum += r;
                }
                iir_x_delay_.Push(left_in, right_in);
                left_sum = left_sum * 0.5f + left_in * iir_fir_k_;
                right_sum = right_sum * 0.5f + right_in * iir_fir_k_;

                auto remove_positive_spectrum = hilbert_complex_.Tick(simd::Float128{
                    left_sum.real(), left_sum.imag(), right_sum.real(), right_sum.imag()
                });
                // this will mirror the positive spectrum to negative domain, forming a real value signal
                auto damp_x = simd::Shuffle<simd::Float128, 0, 2, 1, 3>(remove_positive_spectrum, remove_positive_spectrum);

                *left_ptr = damp_x[0] * wet_mix + left_in * dry_mix;
                *right_ptr = damp_x[1] * wet_mix + right_in * dry_mix;
                ++left_ptr;
                ++right_ptr;
            }

            barber_osc_keep_amp_counter_ += len;
            [[unlikely]]
            if (barber_osc_keep_amp_counter_ > barber_osc_keep_amp_need_) {
                barber_osc_keep_amp_counter_ = 0;
                barber_oscillator_.KeepAmp();
            }
        }
        last_delay_samples_ = last_exp_delay_samples_;
        last_damp_lowpass_coeff_ = damp_lowpass_coeff_;
    }
}
