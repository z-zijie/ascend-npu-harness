#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <random>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <limits>
#include "harness/shape.hpp"
#include "harness/dtype.hpp"

namespace harness {
namespace test {

// Deterministic RNG for reproducible tests
class DeterministicRng {
public:
    explicit DeterministicRng(uint64_t seed = 20260522) : rng_(seed) {}

    float uniform_float(float lo = -1.0f, float hi = 1.0f) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(rng_);
    }

    int uniform_int(int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(rng_);
    }

    uint64_t seed() const { return 20260522; }

private:
    std::mt19937_64 rng_;
};

// Generate random float vector
inline std::vector<float> generate_random_float(size_t n, uint64_t seed = 20260522,
                                                  float lo = -1.0f, float hi = 1.0f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = dist(rng);
    return out;
}

// Generate random int vector
inline std::vector<int32_t> generate_random_int(size_t n, uint64_t seed = 20260522,
                                                  int lo = -100, int hi = 100) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int32_t> dist(lo, hi);
    std::vector<int32_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = dist(rng);
    return out;
}

// Generate sinusoidal cos/sin values for RoPE
inline void generate_rope_cos_sin(size_t seq_len, size_t rotary_dim_half,
                                   std::vector<float>& cos_out,
                                   std::vector<float>& sin_out,
                                   float base = 10000.0f) {
    cos_out.resize(seq_len * rotary_dim_half);
    sin_out.resize(seq_len * rotary_dim_half);
    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < rotary_dim_half; ++d) {
            float theta = static_cast<float>(s) / std::pow(base, 2.0f * d / (rotary_dim_half * 2));
            cos_out[s * rotary_dim_half + d] = std::cos(theta);
            sin_out[s * rotary_dim_half + d] = std::sin(theta);
        }
    }
}

// Convert fp32 vector to fp16 (half precision) representation
inline std::vector<uint16_t> float32_to_float16(const std::vector<float>& src) {
    std::vector<uint16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        float f = src[i];
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (bits >> 13) & 0x3FF;

        if ((bits & 0x7FFFFFFF) == 0) {
            dst[i] = static_cast<uint16_t>(sign);  // zero
        } else if (exp >= 31) {
            dst[i] = static_cast<uint16_t>(sign | 0x7C00);  // inf
        } else if (exp <= 0) {
            dst[i] = static_cast<uint16_t>(sign);  // underflow to zero
        } else {
            dst[i] = static_cast<uint16_t>(sign | (exp << 10) | mant);
        }
    }
    return dst;
}

// Convert fp16 to fp32
inline float float16_to_float32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }
        // Subnormal: treat as zero for simplicity
        uint32_t bits = sign;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    } else if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    exp = exp - 15 + 127;
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline std::vector<float> float16_to_float32(const std::vector<uint16_t>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = float16_to_float32(src[i]);
    }
    return dst;
}

inline std::string seed_string(uint64_t seed = 20260522) {
    std::ostringstream oss;
    oss << "seed=" << seed;
    return oss.str();
}

}  // namespace test
}  // namespace harness
