#pragma once
#include <array>
#include <atomic>
#include <algorithm>
#include <vector>

#include <simd_detector.h>
#include <qwqdsp/extension_marcos.hpp>
#include <qwqdsp/filter/window_fir.hpp>
#include <qwqdsp/misc/smoother.hpp>
#include <qwqdsp/oscillator/vic_sine_osc.hpp>
#include <qwqdsp/simd_element/align_allocator.hpp>
#include <qwqdsp/window/kaiser.hpp>
#include <AudioFFTcpx.h>

#include "global.hpp"
#include "pluginshared/dsp/one_pole_tpt.hpp"
#include "pluginshared/dsp/stereo_iir_hilbert_cpx.hpp"
#include "pluginshared/simd.hpp"

struct Complex32x4 {
    simd::Float128 re;
    simd::Float128 im;

    constexpr Complex32x4& operator*=(const Complex32x4& a) noexcept {
        auto new_re = re * a.re - im * a.im;
        auto new_im = re * a.im + im * a.re;
        re = new_re;
        im = new_im;
        return *this;
    }
};

struct Complex32x8 {
    simd::Float256 re;
    simd::Float256 im;

    constexpr Complex32x8& operator*=(const Complex32x8& a) noexcept {
        simd::Float256 new_re = re * a.re - im * a.im;
        simd::Float256 new_im = re * a.im + im * a.re;
        re = new_re;
        im = new_im;
        return *this;
    }
};

template <simd::IsSimdFloat T>
struct SimdComplex {
    T re;
    T im;

    std::complex<float> ReduceAdd() noexcept {
        return std::complex<float>{simd::ReduceAdd(re), simd::ReduceAdd(im)};
    }

    SimdComplex Conj() noexcept {
        return {re, -im};
    }
};
template <simd::IsSimdFloat T>
static inline SimdComplex<T> operator*(SimdComplex<T> a, SimdComplex<T> b) noexcept {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
template <simd::IsSimdFloat T>
static inline SimdComplex<T> operator+(SimdComplex<T> a, SimdComplex<T> b) noexcept {
    return {a.re + b.re, a.im + b.im};
}
template <simd::IsSimdFloat T>
static inline SimdComplex<T> operator*(T a, SimdComplex<T> b) noexcept {
    return {a * b.re, a * b.im};
}
template <simd::IsSimdFloat T>
static inline SimdComplex<T> operator*(float a, SimdComplex<T> b) noexcept {
    return {a * b.re, a * b.im};
}
template <simd::IsSimdFloat T>
static inline SimdComplex<T> operator*(std::complex<float> a, SimdComplex<T> b) noexcept {
    return {a.real() * b.re - a.imag() * b.im, a.real() * b.im + a.imag() * b.re};
}

class Vec4DelayLine {
public:
    void Init(float max_ms, float fs) {
        float d = max_ms * fs / 1000.0f;
        size_t i = static_cast<size_t>(std::ceil(d));
        Init(i);
    }

    void Init(size_t max_samples) {
        size_t a = 1;
        while (a < max_samples) {
            a *= 2;
        }
        mask_ = static_cast<int>(a - 1);
        delay_length_ = static_cast<int>(a);
        buffer_.resize(a * 2);
    }

    void Reset() noexcept {
        wpos_ = 0;
        std::fill(buffer_.begin(), buffer_.end(), 0.0f);
    }

    simd::Float128 GetAfterPush(simd::Float128 delay_samples) const noexcept {
        simd::Float128 frpos = static_cast<float>(wpos_ + delay_length_) - delay_samples;
        auto t = simd::Frac128(frpos);
        auto rpos = simd::ToInt128(frpos) - 1;
        auto irpos = rpos & mask_;

        simd::Float128 interp0 = simd::Loadu128(buffer_.data() + irpos[0]);
        simd::Float128 interp1 = simd::Loadu128(buffer_.data() + irpos[1]);
        simd::Float128 interp2 = simd::Loadu128(buffer_.data() + irpos[2]);
        simd::Float128 interp3 = simd::Loadu128(buffer_.data() + irpos[3]);
        auto [yn1, y0, y1, y2] = simd::Transpose(interp0, interp1, interp2, interp3);

        auto d0 = (y1 - yn1) * 0.5f;
        auto d1 = (y2 - y0) * 0.5f;
        auto d = y1 - y0;
        auto m0 = 3.0f * d - 2.0f * d0 - d1;
        auto m1 = d0 - 2.0f * d + d1;
        return y0 + t * (d0 + t * (m0 + t * m1));
    }

    simd::Float256 GetAfterPush(simd::Float256 delay_samples) const noexcept {
        auto frpos = static_cast<float>(wpos_ + delay_length_) - delay_samples;
        auto t = simd::Frac256(frpos);
        auto rpos = simd::ToInt256(frpos) - 1;
        auto irpos = rpos & mask_;

        simd::Float128 interp0 = simd::Loadu128(buffer_.data() + irpos[0]);
        simd::Float128 interp1 = simd::Loadu128(buffer_.data() + irpos[1]);
        simd::Float128 interp2 = simd::Loadu128(buffer_.data() + irpos[2]);
        simd::Float128 interp3 = simd::Loadu128(buffer_.data() + irpos[3]);
        simd::Float128 interp4 = simd::Loadu128(buffer_.data() + irpos[4]);
        simd::Float128 interp5 = simd::Loadu128(buffer_.data() + irpos[5]);
        simd::Float128 interp6 = simd::Loadu128(buffer_.data() + irpos[6]);
        simd::Float128 interp7 = simd::Loadu128(buffer_.data() + irpos[7]);

        auto [yn1, y0, y1, y2] =
            simd::Transpose256(interp0, interp1, interp2, interp3, interp4, interp5, interp6, interp7);

        auto d0 = (y1 - yn1) * 0.5f;
        auto d1 = (y2 - y0) * 0.5f;
        auto d = y1 - y0;
        auto m0 = 3.0f * d - 2.0f * d0 - d1;
        auto m1 = d0 - 2.0f * d + d1;
        return y0 + t * (d0 + t * (m0 + t * m1));
    }

    void Push(float x) noexcept {
        wpos_ = (wpos_ + 1) & mask_;
        buffer_[static_cast<size_t>(wpos_)] = x;
        buffer_[static_cast<size_t>(wpos_ + delay_length_)] = x;
    }
private:
    std::vector<float, qwqdsp_simd_element::AlignedAllocator<float, 32>> buffer_;
    int delay_length_{};
    int wpos_{};
    int mask_{};
};

template <simd::IsSimdFloat T>
class ParallelDelayLine {
public:
    struct State {
        T y_re;
        T y_im;
        T y_conj_re;
        T y_conj_im;
    };

    void Init(float max_ms, float fs) {
        float d = max_ms * fs / 1000.0f;
        size_t i = static_cast<size_t>(std::ceil(d));
        Init(i);
    }

    void Init(size_t max_samples) {
        size_t a = 1;
        while (a < max_samples) {
            a *= 2;
        }
        mask_ = static_cast<int>(a - 1);
        delay_length_ = static_cast<int>(a);
        buffer_.resize(a * 2);
    }

    void Reset() noexcept {
        wpos_ = 0;
        std::fill(buffer_.begin(), buffer_.end(), State{});
    }

    State GetBeforePush(float delay_samples) const noexcept {
        float frpos = static_cast<float>(wpos_ + delay_length_) - delay_samples;
        auto t = frpos - std::floor(frpos);
        auto rpos = static_cast<int>(frpos);
        auto irpos = rpos & mask_;
        auto rprev1 = (irpos - 1) & (mask_);
        auto rnext1 = (irpos + 1) & (mask_);
        auto rnext2 = (irpos + 2) & (mask_);

        auto yn1 = buffer_[static_cast<size_t>(rprev1)];
        auto y0 = buffer_[static_cast<size_t>(irpos)];
        auto y1 = buffer_[static_cast<size_t>(rnext1)];
        auto y2 = buffer_[static_cast<size_t>(rnext2)];

        // auto d0 = (y1 - yn1) * 0.5f;
        auto d0_y_re = (y1.y_re - yn1.y_re) * 0.5f;
        auto d0_y_im = (y1.y_im - yn1.y_im) * 0.5f;
        auto d0_y_conj_re = (y1.y_conj_re - yn1.y_conj_re) * 0.5f;
        auto d0_y_conj_im = (y1.y_conj_im - yn1.y_conj_im) * 0.5f;
        // auto d1 = (y2 - y0) * 0.5f;
        auto d1_y_re = (y2.y_re - y0.y_re) * 0.5f;
        auto d1_y_im = (y2.y_im - y0.y_im) * 0.5f;
        auto d1_y_conj_re = (y2.y_conj_re - y0.y_conj_re) * 0.5f;
        auto d1_y_conj_im = (y2.y_conj_im - y0.y_conj_im) * 0.5f;
        // auto d = y1 - y0;
        auto d_y_re = y1.y_re - y0.y_re;
        auto d_y_im = y1.y_im - y0.y_im;
        auto d_y_conj_re = y1.y_conj_re - y0.y_conj_re;
        auto d_y_conj_im = y1.y_conj_im - y0.y_conj_im;
        // auto m0 = 3.0f * d - 2.0f * d0 - d1;
        auto m0_y_re = 3.0f * d_y_re - 2.0f * d0_y_re - d1_y_re;
        auto m0_y_im = 3.0f * d_y_im - 2.0f * d0_y_im - d1_y_im;
        auto m0_y_conj_re = 3.0f * d_y_conj_re - 2.0f * d0_y_conj_re - d1_y_conj_re;
        auto m0_y_conj_im = 3.0f * d_y_conj_im - 2.0f * d0_y_conj_im - d1_y_conj_im;
        // auto m1 = d0 - 2.0f * d + d1;
        auto m1_y_re = d0_y_re - 2.0f * d_y_re + d1_y_re;
        auto m1_y_im = d0_y_im - 2.0f * d_y_im + d1_y_im;
        auto m1_y_conj_re = d0_y_conj_re - 2.0f * d_y_conj_re + d1_y_conj_re;
        auto m1_y_conj_im = d0_y_conj_im - 2.0f * d_y_conj_im + d1_y_conj_im;
        // return y0 + t * (d0 + t * (m0 + t * m1));
        auto y0_y_re = y0.y_re + t * (d0_y_re + t * (m0_y_re + t * m1_y_re));
        auto y0_y_im = y0.y_im + t * (d0_y_im + t * (m0_y_im + t * m1_y_im));
        auto y0_y_conj_re = y0.y_conj_re + t * (d0_y_conj_re + t * (m0_y_conj_re + t * m1_y_conj_re));
        auto y0_y_conj_im = y0.y_conj_im + t * (d0_y_conj_im + t * (m0_y_conj_im + t * m1_y_conj_im));
        return State{y0_y_re, y0_y_im, y0_y_conj_re, y0_y_conj_im};
    }

    void Push(State x) noexcept {
        wpos_ = (wpos_ + 1) & mask_;
        buffer_[static_cast<size_t>(wpos_)] = x;
        buffer_[static_cast<size_t>(wpos_ + delay_length_)] = x;
    }
private:
    std::vector<State> buffer_;
    int delay_length_{};
    int wpos_{};
    int mask_{};
};

class XIirDelayLine {
public:
    void Init(float max_ms, float fs) {
        float d = max_ms * fs / 1000.0f;
        size_t i = static_cast<size_t>(std::ceil(d));
        Init(i);
    }

    void Init(size_t max_samples) {
        size_t a = 1;
        while (a < max_samples) {
            a *= 2;
        }
        mask_ = static_cast<int>(a - 1);
        delay_length_ = static_cast<int>(a);
        buffer_.resize(a * 4);
    }

    void Reset() noexcept {
        wpos_ = 0;
        std::fill(buffer_.begin(), buffer_.end(), float{});
    }

    template<int kChannel>
    float GetBeforePush(float delay_samples) const noexcept {
        if constexpr (kChannel == 0) {
            float frpos = static_cast<float>(wpos_ + delay_length_) - delay_samples;
            auto t = frpos - std::floor(frpos);
            auto rpos = static_cast<int>(frpos);
            auto irpos = rpos & mask_;
            auto rprev1 = (irpos - 1) & (mask_);
            auto rnext1 = (irpos + 1) & (mask_);
            auto rnext2 = (irpos + 2) & (mask_);
    
            auto yn1 = buffer_[static_cast<size_t>(rprev1)];
            auto y0 = buffer_[static_cast<size_t>(irpos)];
            auto y1 = buffer_[static_cast<size_t>(rnext1)];
            auto y2 = buffer_[static_cast<size_t>(rnext2)];
    
            auto d0 = (y1 - yn1) * 0.5f;
            auto d1 = (y2 - y0) * 0.5f;
            auto d = y1 - y0;
            auto m0 = 3.0f * d - 2.0f * d0 - d1;
            auto m1 = d0 - 2.0f * d + d1;
            return y0 + t * (d0 + t * (m0 + t * m1));
        }
        else {
            float frpos = static_cast<float>(wpos_ + delay_length_) - delay_samples;
            auto t = frpos - std::floor(frpos);
            auto rpos = static_cast<int>(frpos);
            auto irpos = rpos & mask_;
            auto rprev1 = (irpos - 1) & (mask_);
            auto rnext1 = (irpos + 1) & (mask_);
            auto rnext2 = (irpos + 2) & (mask_);
    
            auto* ptr = buffer_.data() + delay_length_ * 2;
            auto yn1 = ptr[static_cast<size_t>(rprev1)];
            auto y0 = ptr[static_cast<size_t>(irpos)];
            auto y1 = ptr[static_cast<size_t>(rnext1)];
            auto y2 = ptr[static_cast<size_t>(rnext2)];
    
            auto d0 = (y1 - yn1) * 0.5f;
            auto d1 = (y2 - y0) * 0.5f;
            auto d = y1 - y0;
            auto m0 = 3.0f * d - 2.0f * d0 - d1;
            auto m1 = d0 - 2.0f * d + d1;
            return y0 + t * (d0 + t * (m0 + t * m1));
        }
    }

    void Push(float xleft, float xright) noexcept {
        wpos_ = (wpos_ + 1) & mask_;
        buffer_[static_cast<size_t>(wpos_)] = xleft;
        buffer_[static_cast<size_t>(wpos_ + delay_length_)] = xleft;
        int wpos2 = wpos_ + delay_length_ + delay_length_;
        buffer_[static_cast<size_t>(wpos2)] = xright;
        buffer_[static_cast<size_t>(wpos2 + delay_length_)] = xright;
    }
private:
    std::vector<float> buffer_;
    int delay_length_{};
    int wpos_{};
    int mask_{};
};

template <simd::IsSimdFloat T>
class IirNFilter {
public:
    void Init(float fs, float max_ms) {
        delay_l_.Init(max_ms, fs);
        delay_r_.Init(max_ms, fs);
    }

    void Reset() noexcept {
        delay_l_.Reset();
        delay_r_.Reset();
    }

    std::array<float, 2> Tick(float left, float right, float delay_l, float delay_r) noexcept {
        auto s_l = delay_l_.GetBeforePush(delay_l);
        auto s_r = delay_r_.GetBeforePush(delay_r);

        // auto y = s.y * pole + s.x * residual;
        SimdComplex<T> y_l = SimdComplex<T>{s_l.y_re, s_l.y_im} * pole_ + left * residual_;
        SimdComplex<T> y_r = SimdComplex<T>{s_r.y_re, s_r.y_im} * pole_ + right * residual_;

        delay_l_.Push(typename decltype(delay_l_)::State{y_l.re, y_l.im, T{}, T{}});
        delay_r_.Push(typename decltype(delay_r_)::State{y_r.re, y_r.im, T{}, T{}});
        return {simd::ReduceAdd(y_l.re), simd::ReduceAdd(y_r.re)};
    }

    std::array<std::complex<float>, 2> TickCpx(
        float left, float right,
        float delay_l, float delay_r,
        std::complex<float> zrotate_l, std::complex<float> zrotate_r
    ) noexcept {
        auto s_l = delay_l_.GetBeforePush(delay_l);
        auto s_r = delay_r_.GetBeforePush(delay_r);

        // auto y = s.y * pole + s.x * residual;
        SimdComplex<T> y_l = SimdComplex<T>{s_l.y_re, s_l.y_im} * pole_ + left * residual_;
        SimdComplex<T> y_r = SimdComplex<T>{s_r.y_re, s_r.y_im} * pole_ + right * residual_;
        y_l = zrotate_l * y_l;
        y_r = zrotate_r * y_r;
        SimdComplex<T> y_l_conj = SimdComplex<T>{s_l.y_conj_re, s_l.y_conj_im} * pole_.Conj() + left * residual_.Conj();
        SimdComplex<T> y_r_conj = SimdComplex<T>{s_r.y_conj_re, s_r.y_conj_im} * pole_.Conj() + right * residual_.Conj();
        y_l_conj = zrotate_l * y_l_conj;
        y_r_conj = zrotate_r * y_r_conj;

        delay_l_.Push(typename decltype(delay_l_)::State{y_l.re, y_l.im, y_l_conj.re, y_l_conj.im});
        delay_r_.Push(typename decltype(delay_r_)::State{y_r.re, y_r.im, y_r_conj.re, y_r_conj.im});
        return {(y_l + y_l_conj).ReduceAdd(), (y_r + y_r_conj).ReduceAdd()};
    }

    void Set(SimdComplex<T> residual, SimdComplex<T> pole) noexcept {
        pole_ = pole;
        residual_ = residual;
    }
private:
    ParallelDelayLine<T> delay_l_;
    ParallelDelayLine<T> delay_r_;
    SimdComplex<T> pole_;
    SimdComplex<T> residual_;
};
using Iir4Filter = IirNFilter<simd::Float128>;
using Iir8Filter = IirNFilter<simd::Float256>;

struct SteepFlangerParameter {
    // => is mapping internal
    float delay_ms;       // >=0
    float depth_ms;       // >=0
    float lfo_freq;       // hz
    float lfo_phase;      // 0~1 => 0~2pi
    float fir_cutoff;     // 0~pi
    size_t fir_coeff_len; // 4~kMaxCoeffLen
    float fir_side_lobe;  // >20
    bool fir_min_phase;
    bool fir_highpass;
    float feedback; // gain
    float damp_pitch;
    float barber_phase; // 0~1 => 0~2pi
    float barber_speed; // hz
    bool barber_enable;
    float barber_stereo_phase;              // 0~pi/2
    float drywet;                           // 0~1
    std::atomic<bool> should_update_fir_{}; // tell flanger to update coeffs
    std::atomic<bool> is_using_custom_{};
    std::array<float, global::kMaxCoeffLen> custom_coeffs_{};
    std::array<float, global::kMaxCoeffLen> custom_spectral_gains{};

    // iir mode
    bool iir_mode;
    std::atomic<bool> should_update_iir_{}; // tell flanger to update coeffs
    size_t iir_num_filters{};
    // `iir cutoff` is using `fir_cutoff`
    float ripple{}; // >0
};

class SteepFlanger {
public:
    struct DispatchInfo {
        using DispatchFunction = void (SteepFlanger::*)(float* left_ptr, float* right_ptr, size_t len, SteepFlangerParameter& param) noexcept;

        DispatchFunction dispatch_func{};
        DispatchFunction iir_dispatch_func{};
        size_t lane_size{};

        bool IsValid() const noexcept { return dispatch_func != nullptr; }
    };

    SteepFlanger() {
        complex_fft_.init(global::kFFTSize);

#ifdef VEC4_DISPATCH_INSTRUCTIONS
        if (simd_detector::is_supported(simd_detector::InstructionSet::VEC4_DISPATCH_INSTRUCTIONS)) {
            dispatch_info_.dispatch_func = &SteepFlanger::ProcessVec4;
            dispatch_info_.iir_dispatch_func = &SteepFlanger::ProcessVec4_Iir;
            dispatch_info_.lane_size = 4;
            new (&iir_filters_.iir4) Iir4Filter[global::kIirMaxNumFilters / 4];
        }
#endif
#ifdef VEC4_2_DISPATCH_INSTRUCTIONS
        if (simd_detector::is_supported(simd_detector::InstructionSet::VEC4_2_DISPATCH_INSTRUCTIONS)) {
            dispatch_info_.dispatch_func = &SteepFlanger::ProcessVec4_2;
            dispatch_info_.iir_dispatch_func = &SteepFlanger::ProcessVec4_2_Iir;
            dispatch_info_.lane_size = 4;
            new (&iir_filters_.iir4) Iir4Filter[global::kIirMaxNumFilters / 4];
        }
#endif
#ifdef VEC8_DISPATCH_INSTRUCTIONS
        if (simd_detector::is_supported(simd_detector::InstructionSet::VEC8_DISPATCH_INSTRUCTIONS)) {
            dispatch_info_.dispatch_func = &SteepFlanger::ProcessVec8;
            dispatch_info_.iir_dispatch_func = &SteepFlanger::ProcessVec8_Iir;
            dispatch_info_.lane_size = 8;
            new (&iir_filters_.iir8) Iir8Filter[global::kIirMaxNumFilters / 8];
        }
#endif
#ifdef VEC8_2_DISPATCH_INSTRUCTIONS
        if (simd_detector::is_supported(simd_detector::InstructionSet::VEC8_2_DISPATCH_INSTRUCTIONS)) {
            dispatch_info_.dispatch_func = &SteepFlanger::ProcessVec8_2;
            dispatch_info_.iir_dispatch_func = &SteepFlanger::ProcessVec8_2_Iir;
            dispatch_info_.lane_size = 8;
            new (&iir_filters_.iir8) Iir8Filter[global::kIirMaxNumFilters / 8];
        }
#endif
    }

    ~SteepFlanger() {
        if (dispatch_info_.lane_size == 4) {
            for (size_t i = 0; i < global::kIirMaxNumFilters / 4; ++i) {
                iir_filters_.iir4[i].~Iir4Filter();
            }
        }
        else if (dispatch_info_.lane_size == 8) {
            for (size_t i = 0; i < global::kIirMaxNumFilters / 8; ++i) {
                iir_filters_.iir8[i].~Iir8Filter();
            }
        }
    }

    void Init(float fs, float max_delay_ms) {
        if (!dispatch_info_.IsValid()) return;

        if (dispatch_info_.lane_size == 4) {
            for (size_t i = 0; i < global::kIirMaxNumFilters / 4; ++i) {
                iir_filters_.iir4[i].Init(fs, max_delay_ms);
            }
        }
        else if (dispatch_info_.lane_size == 8) {
            for (size_t i = 0; i < global::kIirMaxNumFilters / 8; ++i) {
                iir_filters_.iir8[i].Init(fs, max_delay_ms);
            }
        }
        else {
            assert(false);
        }
        iir_x_delay_.Init(fs, max_delay_ms);

        fs_ = fs;
        float const samples_need = fs * max_delay_ms / 1000.0f;
        delay_left_.Init(static_cast<size_t>(samples_need * global::kMaxCoeffLen));
        delay_right_.Init(static_cast<size_t>(samples_need * global::kMaxCoeffLen));
        barber_phase_smoother_.SetSmoothTime(20.0f, fs);
        damp_.Reset();
        barber_oscillator_.Reset();
        barber_osc_keep_amp_counter_ = 0;
        // VIC正交振荡器衰减非常慢，设定为5分钟保持一次
        barber_osc_keep_amp_need_ = static_cast<size_t>(fs * 60 * 5);
    }

    void Reset() noexcept {
        delay_left_.Reset();
        delay_right_.Reset();
        left_fb_ = 0;
        right_fb_ = 0;
        damp_.Reset();
        hilbert_complex_.Reset();
        if (dispatch_info_.lane_size == 4) {
            for (size_t i = 0; i < global::kIirMaxNumFilters / 4; ++i) {
                iir_filters_.iir4[i].Reset();
            }
        }
        else if (dispatch_info_.lane_size == 8) {
            for (size_t i = 0; i < global::kIirMaxNumFilters / 8; ++i) {
                iir_filters_.iir8[i].Reset();
            }
        }
    }

    void Process(float* left_ptr, float* right_ptr, size_t len, SteepFlangerParameter& param) noexcept {
        if (dispatch_info_.IsValid()) {
            if (!param.iir_mode) {
                (this->*dispatch_info_.dispatch_func)(left_ptr, right_ptr, len, param);
            }
            else {
                (this->*dispatch_info_.iir_dispatch_func)(left_ptr, right_ptr, len, param);
            }
        }
    }

    /**
     * @param p [0, 1]
     */
    void SetLFOPhase(float p) noexcept {
        phase_ = p;
    }

    /**
     * @param p [0, 1]
     */
    void SetBarberLFOPhase(float p) noexcept {
        barber_oscillator_.Reset(p * std::numbers::pi_v<float> * 2);
    }

    // -------------------- lookup --------------------
    std::span<const float> GetUsingCoeffs() const noexcept {
        return {coeffs_.data(), coeff_len_};
    }

    DispatchInfo GetDispatchInfo() const noexcept {
        return dispatch_info_;
    }

    std::atomic<bool> have_new_coeff_{};
private:
    void ProcessVec8(float* left_ptr, float* right_ptr, size_t len, SteepFlangerParameter& param) noexcept;
    void ProcessVec4(float* left_ptr, float* right_ptr, size_t len, SteepFlangerParameter& param) noexcept;
    void ProcessVec8_2(float* left_ptr, float* right_ptr, size_t len, SteepFlangerParameter& param) noexcept;
    void ProcessVec4_2(float* left_ptr, float* right_ptr, size_t len, SteepFlangerParameter& param) noexcept;

    void ProcessVec8_Iir(float* left_ptr, float* right_ptr, size_t len, SteepFlangerParameter& param) noexcept;
    void ProcessVec4_Iir(float* left_ptr, float* right_ptr, size_t len, SteepFlangerParameter& param) noexcept;
    void ProcessVec8_2_Iir(float* left_ptr, float* right_ptr, size_t len, SteepFlangerParameter& param) noexcept;
    void ProcessVec4_2_Iir(float* left_ptr, float* right_ptr, size_t len, SteepFlangerParameter& param) noexcept;

    void UpdateCoeff(SteepFlangerParameter& param) noexcept {
        size_t coeff_len = static_cast<size_t>(param.fir_coeff_len);
        coeff_len_ = coeff_len;

        if (!param.is_using_custom_) {
            std::span<float> kernel{coeffs_.data(), coeff_len};
            float const cutoff_w = param.fir_cutoff;
            if (param.fir_highpass) {
                qwqdsp_filter::WindowFIR::Highpass(kernel, std::numbers::pi_v<float> - cutoff_w);
            }
            else {
                qwqdsp_filter::WindowFIR::Lowpass(kernel, cutoff_w);
            }
            float const beta = qwqdsp_window::Kaiser::Beta(param.fir_side_lobe);
            qwqdsp_window::Kaiser::ApplyWindow(kernel, beta, false);
        }
        else {
            std::copy_n(param.custom_coeffs_.begin(), coeff_len, coeffs_.begin());
        }

        if (dispatch_info_.lane_size == 4) {
            size_t const coeff_len_div_4 = (coeff_len + 3) / 4;
            size_t const idxend = coeff_len_div_4 * 4;
            for (size_t i = coeff_len; i < idxend; ++i) {
                coeffs_[i] = 0;
            }
        }
        else if (dispatch_info_.lane_size == 8) {
            size_t const coeff_len_div_8 = (coeff_len + 7) / 8;
            size_t const idxend = coeff_len_div_8 * 8;
            for (size_t i = coeff_len; i < idxend; ++i) {
                coeffs_[i] = 0;
            }
        }

        std::span<float> kernel{coeffs_.data(), coeff_len};
        constexpr size_t num_bins = audiofft::AudioFFTcpx::ComplexSize(global::kFFTSize);
        float pad[global::kFFTSize]{};
        float pad_im[global::kFFTSize]{};
        std::array<float, num_bins> gains{};
        std::array<float, num_bins> fft_re{};
        std::array<float, num_bins> fft_im{};
        std::copy(kernel.begin(), kernel.end(), pad);

        complex_fft_.fft(pad, pad_im, fft_re.data(), fft_im.data());
        for (size_t i = 0; i < num_bins; ++i) {
            float g = std::sqrt(fft_re[i] * fft_re[i] + fft_im[i] * fft_im[i]);
            gains[i] = g;
        }

        if (param.fir_min_phase) {
            float log_gains[num_bins]{};
            for (size_t i = 0; i < num_bins; ++i) {
                float g = gains[i];
                log_gains[i] = std::log(g + 1e-18f);
            }

            float phases[num_bins]{};
            complex_fft_.ifft(pad, pad_im, log_gains, phases);
            pad[0] = 0;
            pad[num_bins / 2] = 0;
            for (size_t i = num_bins / 2 + 1; i < num_bins; ++i) {
                pad[i] = -pad[i];
            }

            std::fill_n(pad_im, num_bins, 0.0f);
            complex_fft_.fft(pad, pad_im, log_gains, phases);
            for (size_t i = 0; i < num_bins; ++i) {
                fft_re[i] = gains[i] * std::cos(phases[i]);
                fft_im[i] = gains[i] * std::sin(phases[i]);
            }
            complex_fft_.ifft(pad, pad_im, fft_re.data(), fft_im.data());

            for (size_t i = 0; i < kernel.size(); ++i) {
                kernel[i] = pad[i];
            }
        }

        float const max_spectral_gain = *std::max_element(gains.begin(), gains.end());
        float gain = 1.0f / (max_spectral_gain + 1e-10f);
        if (max_spectral_gain < 1e-10f) {
            gain = 1.0f;
        }
        for (auto& x : kernel) {
            x *= gain;
        }

        float energy = 0;
        for (auto x : kernel) {
            energy += x * x;
        }
        fir_gain_ = 1.0f / std::sqrt(energy + 1e-10f);

        have_new_coeff_ = true;
    }

    static constexpr size_t kSIMDMaxCoeffLen = ((global::kMaxCoeffLen + 7) / 8) * 8;
    static constexpr float kDelaySmoothMs = 20.0f;

    DispatchInfo dispatch_info_{};
    float fs_{};

    // ----------------------------------------
    // fir part
    // ----------------------------------------
    Vec4DelayLine delay_left_;
    Vec4DelayLine delay_right_;
    simd::Array256<float, kSIMDMaxCoeffLen> coeffs_{};
    simd::Array256<float, kSIMDMaxCoeffLen> last_coeffs_{};
    float fir_gain_{1.0f};
    size_t coeff_len_{};
    // feedback
    float left_fb_{};
    float right_fb_{};
    pluginshared::dsp::OnePoleTPT<simd::Float128> damp_;
    pluginshared::dsp::OnePoleTPT<simd::Float128> dc_;
    float damp_lowpass_coeff_{1.0f};
    float last_damp_lowpass_coeff_{1.0f};

    // ----------------------------------------
    // iir part
    // ----------------------------------------
    union IirFilters {
        IirFilters() {}
        ~IirFilters() {}

        Iir4Filter iir4[global::kIirMaxNumFilters / 4];
        Iir8Filter iir8[global::kIirMaxNumFilters / 8];
    };
    IirFilters iir_filters_;
    float iir_fir_k_{};
    bool last_iir_highpass_{false};
    XIirDelayLine iir_x_delay_;

    // delay time lfo
    float phase_{};
    simd::Float128 last_exp_delay_samples_{};
    simd::Float128 last_delay_samples_{};

    // barberpole
    pluginshared::dsp::StereoIIRHilbertDeeperCpx hilbert_complex_;
    qwqdsp_misc::ExpSmoother barber_phase_smoother_;
    qwqdsp_oscillator::VicSineOsc barber_oscillator_;
    size_t barber_osc_keep_amp_counter_{};
    size_t barber_osc_keep_amp_need_{};

    audiofft::AudioFFTcpx complex_fft_;
};
