#pragma once
#include "decoder.h"
#include "../loggers/tensor_logger.h"
#include "../loggers/ml_logger.h"
#include <cstdint>
#include <vector>

class FlashBPBase;

// ---------------------------------------------------------------------------
// TensorDecoder — generalised min-sum BP with a `degree` hyperparameter.
//
// For each check c, build a tensor over the data nodes in its degree-d
// Tanner-distance neighbourhood:
//
//   N_1(c) = data nodes connected to c
//   N_d(c) = data nodes at Tanner-graph distance 1, 3, …, 2d−1 from c
//
// Each tensor element stores
//   weight[idx] : sum of min-sum costs for binary configuration idx
//   parity[idx] : 0 if every check whose N(c') is fully contained in N_d(c)
//                 is satisfied by configuration idx, else 1
//
// Outgoing edge messages are obtained by marginalisation: for each outgoing
// edge (c, v) with v in N(c), let l_v be the incoming variable→check message;
//
//   W_b = min { weight[idx] − cost(b, l_v) : x_v = b, parity[idx] = 0 }
//   m_{c→v} = W_1 − W_0          (LLR convention: + favours x = 0)
//
// degree = 1 reduces to standard min-sum (same answers as SimpleDecoder).
//
// Memory: per check stores a (weight, parity) tensor of length 2^|N_d(c)|.
// We refuse to construct if any neighbourhood exceeds MAX_AXES.
// ---------------------------------------------------------------------------
template<typename LoggerT>
class TensorDecoder : public Decoder {
public:
    static constexpr int MAX_AXES = 22;   // 2^22 doubles ≈ 32 MB per check

    explicit TensorDecoder(const FlashBPBase& bp,
                           LoggerT            logger,
                           int                degree);

    std::vector<uint8_t> operator()(
        const std::vector<uint8_t>& syndrome,
        int max_iter = 100
    ) override;

    int num_detectors() const override { return num_detectors_; }
    int num_errors()    const override { return num_errors_; }

private:
    int num_detectors_;
    int num_errors_;
    int degree_;

    // ── Tanner-graph structure (same shape as SimpleDecoder) ────────────────
    struct Edge { int check, var; };
    std::vector<Edge>             edges_;
    std::vector<std::vector<int>> var_edges_;
    std::vector<std::vector<int>> check_edges_;
    std::vector<double>           ch_llr_;

    // ── Per-check tensor structure ──────────────────────────────────────────
    // For each check c: sorted list of unique data-node indices in N_d(c).
    std::vector<std::vector<int>> nbhd_data_;
    // For each check c, axis k: the edge index when nbhd_data_[c][k] is in
    // N(c) (direct neighbour); otherwise -1 (deep neighbour, use channel LLR).
    std::vector<std::vector<int>> axis_edge_;

    // For each check c: every other check c' whose N(c') ⊆ N_d(c). For each
    // such c', the bitmask over N_d(c)'s axes that selects N(c'), and c''s
    // own check index (so we can look up its syndrome bit at decode time).
    struct InternalCheck { uint32_t mask; int check_idx; };
    std::vector<std::vector<InternalCheck>> internal_checks_;

    // For each edge ei: axis position of edges_[ei].var inside
    // nbhd_data_[edges_[ei].check].
    std::vector<int> edge_axis_pos_;

    LoggerT      logger_;
    unsigned int shot_counter_ = 0;
};

extern template class TensorDecoder<Logger<false>>;
extern template class TensorDecoder<Logger<true>>;
extern template class TensorDecoder<DecodeLogger<true>>;
extern template class TensorDecoder<RecordLogger>;
extern template class TensorDecoder<TensorLogger>;
extern template class TensorDecoder<MLLogger>;
