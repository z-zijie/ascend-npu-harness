#include "flash_attention_fwd_fused_golden.hpp"
#include <cmath>
#include <cstring>
#include <cfloat>
#include <algorithm>

namespace harness {
namespace flash_attention_fwd_fused {

void flash_attention_fwd_golden(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 float* __restrict__ O,
                                 int64_t B, int64_t H,
                                 int64_t S_q, int64_t S_k,
                                 int64_t D, int64_t D_v,
                                 float scale,
                                 AttnMaskType mask_type) {
    // Online softmax — numerically stable, tile-by-tile over keys
    // Output O is zero-initialized first, then accumulated

    // Zero the output
    size_t total_o = static_cast<size_t>(B * H * S_q * D_v);
    std::memset(O, 0, total_o * sizeof(float));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t q_idx = 0; q_idx < S_q; ++q_idx) {
                // Pointer to current query row
                const float* Q_bhq = Q + ((b * H + h) * S_q + q_idx) * D;

                // Online softmax state for this query
                double m = -1e30;  // running max (float64 for precision)
                double l = 0.0;     // running sum_exp
                // Running output accumulator (float64 for precision)
                double* acc = new double[D_v]();
                // We'll use a stack-allocated buffer when D_v is small,
                // but for large D_v we need heap allocation

                // Process keys in tiles
                int64_t tile_k = 128;  // key tile size
                for (int64_t k_start = 0; k_start < S_k; k_start += tile_k) {
                    int64_t k_end = std::min(k_start + tile_k, S_k);
                    int64_t num_k = k_end - k_start;

                    // Compute scores = Q_bhq @ K^T * scale for this tile
                    // scores[j] = sum_d(Q_bhq[d] * K[b,h,k_start+j,d]) * scale
                    double* scores = new double[num_k];
                    for (int64_t j = 0; j < num_k; ++j) {
                        int64_t k_idx = k_start + j;
                        const float* K_bhk = K + ((b * H + h) * S_k + k_idx) * D;

                        double dot = 0.0;
                        for (int64_t d = 0; d < D; ++d) {
                            dot += static_cast<double>(Q_bhq[d]) *
                                   static_cast<double>(K_bhk[d]);
                        }
                        scores[j] = dot * static_cast<double>(scale);

                        // Apply mask
                        if (mask_type == AttnMaskType::Causal) {
                            // Causal: query i attends to keys j <= i
                            if (k_idx > q_idx) {
                                scores[j] = -1e30;  // effectively -inf
                            }
                        }
                    }

                    // Update online softmax
                    // m_new = max(m, max(scores))
                    double m_old = m;
                    double m_new = m_old;
                    for (int64_t j = 0; j < num_k; ++j) {
                        if (scores[j] > m_new) m_new = scores[j];
                    }

                    // l_new = l * exp(m_old - m_new) + sum(exp(scores - m_new))
                    double l_new = l * std::exp(m_old - m_new);
                    for (int64_t j = 0; j < num_k; ++j) {
                        l_new += std::exp(scores[j] - m_new);
                    }

                    // acc_new = acc * exp(m_old - m_new) + sum_j(exp(scores[j] - m_new) * V[j])
                    double rescale = std::exp(m_old - m_new);
                    for (int64_t dv = 0; dv < D_v; ++dv) {
                        double acc_val = acc[dv] * rescale;
                        for (int64_t j = 0; j < num_k; ++j) {
                            int64_t k_idx = k_start + j;
                            const float* V_bhk = V + ((b * H + h) * S_k + k_idx) * D_v;
                            acc_val += std::exp(scores[j] - m_new) *
                                       static_cast<double>(V_bhk[dv]);
                        }
                        acc[dv] = acc_val;
                    }

                    m = m_new;
                    l = l_new;

                    delete[] scores;
                }

                // Finalize: O = acc / l
                float* O_bhq = O + ((b * H + h) * S_q + q_idx) * D_v;
                for (int64_t dv = 0; dv < D_v; ++dv) {
                    O_bhq[dv] = static_cast<float>(acc[dv] / l);
                }

                delete[] acc;
            }
        }
    }
}

}  // namespace flash_attention_fwd_fused
}  // namespace harness
