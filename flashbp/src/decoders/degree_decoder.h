#pragma once
#include "decoder.h"
#include "../loggers/tensor_logger.h"
#include "../loggers/ml_logger.h"
#include <cstdint>
#include <vector>

class FlashBPBase;

// ---------------------------------------------------------------------------
// DegreeDecoder — generalised min-sum BP centred on parity (check) nodes.
//
// Same outer BP loop as SimpleDecoder:
//   - check-node update produces m_{c->v} for every Tanner edge (c, v)
//   - variable-node update produces extrinsic m_{v->c}
//   - hard decision + convergence check each iteration
//
// The only thing that changes versus standard BP is the *method* for computing
// each check's outgoing messages.  Instead of the standard min-sum formula on
// N(c), we enumerate the joint configuration of the degree-d data-node
// neighbourhood:
//
//   N_d(c) = data nodes at Tanner-distance 1, 3, …, 2d-1 from c
//   weight[x] = Σ_k minsum_cost(x_k, l_k)
//                where l_k = m_{v_k -> c} if v_k in N(c), else channel prior
//   parity[x] = 1 iff some internal check (c' with N(c') ⊆ N_d(c)) is
//               violated by configuration x
//   m_{c -> v} = W_1 - W_0   where
//     W_b = min { weight[x] - minsum_cost(b, l_v) : x_v = b, parity[x] = 0 }
//
// No FVS, no tree-BP, no inter-region reasoning — just a direct enumeration
// of each check's neighbourhood per BP iteration.  Degree-1 recovers the
// standard min-sum equation exactly.
//
// Memory bound: each check builds a (weight, parity) array of length
// 2^|N_d(c)|.  We refuse to construct if any |N_d(c)| > MAX_AXES.
// ---------------------------------------------------------------------------
template<typename LoggerT>
class DegreeDecoder : public Decoder {
public:
    static constexpr int MAX_AXES = 22;

    explicit DegreeDecoder(const FlashBPBase& bp, LoggerT logger, int degree);

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

    // ── Tanner-graph structure ──────────────────────────────────────────────
    struct Edge { int check, var; };
    std::vector<Edge>             edges_;
    std::vector<std::vector<int>> var_edges_;
    std::vector<std::vector<int>> check_edges_;
    std::vector<double>           ch_llr_;

    // ── Per-check tensor structure ──────────────────────────────────────────
    // Sorted list of unique data-node indices forming N_d(c).
    std::vector<std::vector<int>> nbhd_data_;
    // For each axis k in N_d(c): edge index when the axis variable is in N(c)
    // (and so receives a v->c message); -1 otherwise (use channel prior).
    std::vector<std::vector<int>> axis_edge_;

    // Internal checks: every check c' with N(c') ⊆ N_d(c).  Stored as a mask
    // over N_d(c)'s axes plus c''s own detector index (so we can look up its
    // syndrome bit at decode time).
    struct InternalCheck { uint32_t mask; int check_idx; };
    std::vector<std::vector<InternalCheck>> internal_checks_;

    // For each Tanner edge ei: axis position of edges_[ei].var inside
    // nbhd_data_[edges_[ei].check].
    std::vector<int> edge_axis_pos_;

    LoggerT      logger_;
    unsigned int shot_counter_ = 0;
};

extern template class DegreeDecoder<Logger<false>>;
extern template class DegreeDecoder<Logger<true>>;
extern template class DegreeDecoder<DecodeLogger<true>>;
extern template class DegreeDecoder<RecordLogger>;
extern template class DegreeDecoder<TensorLogger>;
extern template class DegreeDecoder<MLLogger>;
