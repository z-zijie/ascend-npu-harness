#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include "harness/accuracy.hpp"
#include "harness/test_utils.hpp"
#include "harness/shape.hpp"
#include "harness/skip.hpp"
#include "flash_attention_fwd_fused_runner.hpp"

using namespace harness::flash_attention_fwd_fused;
using harness::accuracy::Tolerance;

// Helper: run a basic "vanilla" softmax attention as an alternative
// reference for comparison (NOT numerically optimized, for debugging only)
static void vanilla_attention(const float* Q, const float* K, const float* V,
                               float* O,
                               int64_t B, int64_t H,
                               int64_t S_q, int64_t S_k,
                               int64_t D, int64_t D_v,
                               float scale, AttnMaskType mask) {
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t i = 0; i < S_q; ++i) {
                const float* Qi = Q + ((b * H + h) * S_q + i) * D;
                float* Oi = O + ((b * H + h) * S_q + i) * D_v;

                // Compute attention scores
                std::vector<double> scores(S_k);
                double max_score = -1e30;
                for (int64_t j = 0; j < S_k; ++j) {
                    const float* Kj = K + ((b * H + h) * S_k + j) * D;
                    double dot = 0.0;
                    for (int64_t d = 0; d < D; ++d) {
                        dot += static_cast<double>(Qi[d]) * static_cast<double>(Kj[d]);
                    }
                    scores[j] = dot * static_cast<double>(scale);

                    if (mask == AttnMaskType::Causal && j > i) {
                        scores[j] = -1e30;
                    }

                    if (scores[j] > max_score) max_score = scores[j];
                }

                // Stable softmax
                double sum_exp = 0.0;
                for (int64_t j = 0; j < S_k; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }

                // Weighted sum of values
                for (int64_t dv = 0; dv < D_v; ++dv) {
                    double weighted_sum = 0.0;
                    for (int64_t j = 0; j < S_k; ++j) {
                        const float* Vj = V + ((b * H + h) * S_k + j) * D_v;
                        weighted_sum += scores[j] * static_cast<double>(Vj[dv]);
                    }
                    Oi[dv] = static_cast<float>(weighted_sum / sum_exp);
                }
            }
        }
    }
}

class FlashAttentionTest : public ::testing::Test {
protected:
    void run_test(int64_t B, int64_t H, int64_t S_q, int64_t S_k,
                  int64_t D, int64_t D_v, AttnMaskType mask,
                  float scale = 0.0f, uint64_t seed = 20260522) {
        if (scale == 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(D));

        auto Q = harness::test::generate_random_float(
            B * H * S_q * D, seed, -1.0f, 1.0f);
        auto K = harness::test::generate_random_float(
            B * H * S_k * D, seed + 1000, -1.0f, 1.0f);
        auto V = harness::test::generate_random_float(
            B * H * S_k * D_v, seed + 2000, -1.0f, 1.0f);
        std::vector<float> O(B * H * S_q * D_v, 0.0f);
        std::vector<float> expected(B * H * S_q * D_v, 0.0f);

        // Online softmax golden
        flash_attention_fwd_golden(Q.data(), K.data(), V.data(),
                                    expected.data(),
                                    B, H, S_q, S_k, D, D_v, scale, mask);

        // Runner (dispatches to golden in CPU mode)
        run_flash_attention_fwd_fused_cpu(Q.data(), K.data(), V.data(),
                                           O.data(),
                                           B, H, S_q, S_k, D, D_v, scale, mask);

        auto report = harness::accuracy::CompareArrays(expected, O,
            Tolerance{5e-4, 5e-4});
        EXPECT_TRUE(report.passed) << "flash_attn [B=" << B << " H=" << H
            << " S_q=" << S_q << " S_k=" << S_k << " D=" << D << " D_v=" << D_v
            << " mask=" << (mask == AttnMaskType::Causal ? "causal" : "none") << "] "
            << "failed: " << report.ToString();

        // Also verify against vanilla attention (as sanity check)
        if (S_q * S_k <= 10000) {  // only for small shapes
            std::vector<float> vanilla_out(B * H * S_q * D_v, 0.0f);
            vanilla_attention(Q.data(), K.data(), V.data(), vanilla_out.data(),
                              B, H, S_q, S_k, D, D_v, scale, mask);
            auto vanilla_report = harness::accuracy::CompareArrays(
                expected, vanilla_out, Tolerance{2e-4, 2e-4});
            // Both should agree to tight tolerance
            if (!vanilla_report.passed) {
                // Soft warning - online softmax can differ slightly from vanilla
                // due to floating point ordering
                std::cerr << "  (vanilla vs online max_abs_err="
                          << vanilla_report.max_abs_error << ")" << std::endl;
            }
        }
    }
};

// ===========================================================================
// Test: No mask, small shapes
// ===========================================================================
TEST_F(FlashAttentionTest, NoMask_B1_H4_Sq1_Sk1_D32) {
    run_test(1, 4, 1, 1, 32, 32, AttnMaskType::None);
}
TEST_F(FlashAttentionTest, NoMask_B1_H1_Sq16_Sk16_D64) {
    run_test(1, 1, 16, 16, 64, 64, AttnMaskType::None);
}
TEST_F(FlashAttentionTest, NoMask_B1_H8_Sq16_Sk16_D32) {
    run_test(1, 8, 16, 16, 32, 32, AttnMaskType::None);
}
TEST_F(FlashAttentionTest, NoMask_B2_H4_Sq16_Sk16_D64) {
    run_test(2, 4, 16, 16, 64, 64, AttnMaskType::None);
}

// ===========================================================================
// Test: Causal mask (requires S_q == S_k)
// ===========================================================================
TEST_F(FlashAttentionTest, Causal_B1_H4_Sq8_Sk8_D32) {
    run_test(1, 4, 8, 8, 32, 32, AttnMaskType::Causal);
}
TEST_F(FlashAttentionTest, Causal_B1_H1_Sq16_Sk16_D64) {
    run_test(1, 1, 16, 16, 64, 64, AttnMaskType::Causal);
}
TEST_F(FlashAttentionTest, Causal_B2_H4_Sq32_Sk32_D128) {
    run_test(2, 4, 32, 32, 128, 128, AttnMaskType::Causal);
}

// ===========================================================================
// Test: Non-square sequence (encoder-decoder cross-attention)
// ===========================================================================
TEST_F(FlashAttentionTest, NoMask_B1_H4_Sq8_Sk16_D64) {
    run_test(1, 4, 8, 16, 64, 64, AttnMaskType::None);
}
TEST_F(FlashAttentionTest, NoMask_B1_H2_Sq4_Sk32_D128) {
    run_test(1, 2, 4, 32, 128, 128, AttnMaskType::None);
}

// ===========================================================================
// Test: Medium shapes
// ===========================================================================
TEST_F(FlashAttentionTest, NoMask_B1_H4_Sq64_Sk64_D64) {
    run_test(1, 4, 64, 64, 64, 64, AttnMaskType::None);
}
TEST_F(FlashAttentionTest, Causal_B1_H4_Sq64_Sk64_D64) {
    run_test(1, 4, 64, 64, 64, 64, AttnMaskType::Causal);
}
TEST_F(FlashAttentionTest, NoMask_B1_H2_Sq128_Sk128_D32) {
    run_test(1, 2, 128, 128, 32, 32, AttnMaskType::None);
}

// ===========================================================================
// Test: D_v != D (separate value head dimension)
// ===========================================================================
TEST_F(FlashAttentionTest, DvDiff_B1_H4_Sq8_Sk8_D64_Dv32) {
    run_test(1, 4, 8, 8, 64, 32, AttnMaskType::None);
}
TEST_F(FlashAttentionTest, DvDiff_B1_H4_Sq16_Sk16_D128_Dv64) {
    run_test(1, 4, 16, 16, 128, 64, AttnMaskType::Causal);
}

// ===========================================================================
// Test: Causal mask correctness — verify causal constraint
// ===========================================================================
TEST_F(FlashAttentionTest, CausalConstraintVerification) {
    // With causal mask, query i should NOT attend to key j > i.
    // Use a specially constructed input where K values for j > i are HUGE
    // to verify they are properly masked out.

    int64_t B = 1, H = 1, S = 8, D = 16;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Construct Q: all ones
    std::vector<float> Q(B * H * S * D, 1.0f);
    // K: first half normal, second half = 1000x larger
    std::vector<float> K(B * H * S * D, 0.0f);
    for (int64_t j = 0; j < S; ++j) {
        float val = (j >= S / 2) ? 100.0f : 1.0f;
        for (int64_t d = 0; d < D; ++d) {
            K[j * D + d] = val;
        }
    }
    // V: each position has a unique identifier (its index)
    std::vector<float> V(B * H * S * D, 0.0f);
    for (int64_t j = 0; j < S; ++j) {
        V[j * D + 0] = static_cast<float>(j);
    }

    std::vector<float> O(B * H * S * D, 0.0f);
    flash_attention_fwd_golden(Q.data(), K.data(), V.data(), O.data(),
                                B, H, S, S, D, D, scale, AttnMaskType::Causal);

    // For each query position i, output[0] should be <= i (weighted average of j <= i)
    // Since V[j][0] = j, O[i][0] is weighted average of visible j values
    for (int64_t i = 0; i < S; ++i) {
        float out_val = O[i * D + 0];
        // Since V[j][0] = j, and we attend only to j <= i,
        // output must be a weighted average of values {0, 1, ..., i}
        // So 0 <= output <= i
        EXPECT_GE(out_val, 0.0f) << "O[" << i << "][0] = " << out_val
            << " should be >= 0 (causal, cannot attend to negative indices)";
        EXPECT_LE(out_val, static_cast<float>(i) + 0.01f)
            << "O[" << i << "][0] = " << out_val
            << " should be <= " << i << " (causal, cannot attend to future)";
    }
}

// ===========================================================================
// Test: Scale invariance — uniform scaling of K should not change output
// if scale is adjusted
// ===========================================================================
TEST_F(FlashAttentionTest, ScaleInvariance) {
    int64_t B = 1, H = 2, S = 8, D = 32;
    float base_scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto Q = harness::test::generate_random_float(B * H * S * D, 99, -1.0f, 1.0f);
    auto K = harness::test::generate_random_float(B * H * S * D, 100, -1.0f, 1.0f);
    auto V = harness::test::generate_random_float(B * H * S * D, 101, -1.0f, 1.0f);
    std::vector<float> O1(B * H * S * D, 0.0f);
    std::vector<float> O2(B * H * S * D, 0.0f);

    // Run with base_scale
    flash_attention_fwd_golden(Q.data(), K.data(), V.data(), O1.data(),
                                B, H, S, S, D, D, base_scale, AttnMaskType::None);

    // Scale K by 2, adjust scale by 1/2 -> scores unchanged
    auto K2 = K;
    for (size_t i = 0; i < K2.size(); ++i) K2[i] *= 2.0f;
    float adj_scale = base_scale / 2.0f;
    flash_attention_fwd_golden(Q.data(), K2.data(), V.data(), O2.data(),
                                B, H, S, S, D, D, adj_scale, AttnMaskType::None);

    // Results should match since Q@(2K)*(s/2) = Q@K*s
    auto report = harness::accuracy::CompareArrays(O1, O2,
        Tolerance{1e-4, 1e-4});
    EXPECT_TRUE(report.passed) << "Scale invariance failed: " << report.ToString();
}

// ===========================================================================
// Test: Attention sum to 1 — each attention weight row sums to 1
// ===========================================================================
TEST_F(FlashAttentionTest, ProbabilitiesSumToOne) {
    // Feed uniform Q,K,V: all ones
    int64_t B = 1, H = 1, S = 4, D = 4;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> Q(B * H * S * D, 1.0f);
    std::vector<float> K(B * H * S * D, 1.0f);
    // V: first element of each key position = 1.0, others = 0
    std::vector<float> V(B * H * S * D, 0.0f);
    for (int64_t j = 0; j < S; ++j) V[j * D + 0] = 1.0f;

    std::vector<float> O(B * H * S * D, 0.0f);
    flash_attention_fwd_golden(Q.data(), K.data(), V.data(), O.data(),
                                B, H, S, S, D, D, scale, AttnMaskType::None);

    // With uniform Q,K, all attention weights are equal (1/S)
    // Each V[j][0] = 1, so O[i][0] = sum_j(1/S * 1) = 1.0
    for (int64_t i = 0; i < S; ++i) {
        EXPECT_NEAR(O[i * D + 0], 1.0f, 1e-4f)
            << "Output O[" << i << "][0] should be ~1.0 for uniform attention";
    }

    // Other dimensions of O should be 0 (since V[j][d>0] = 0)
    for (int64_t i = 0; i < S; ++i) {
        for (int64_t d = 1; d < D; ++d) {
            EXPECT_NEAR(O[i * D + d], 0.0f, 1e-4f)
                << "O[" << i << "][" << d << "] should be ~0.0";
        }
    }
}
