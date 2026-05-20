#pragma once
#include "decoder.h"
#include "../loggers/tensor_logger.h"
#include "../loggers/ml_logger.h"
#include <cstdint>
#include <vector>

class FlashBPBase;

// ---------------------------------------------------------------------------
// FvsDecoder — degree-d generalised BP with a Feedback Vertex Set.
//
// For each check c:
//   N_d(c)      = data nodes at Tanner-distance 1, 3, ..., 2d-1 from c
//   S_d(c)      = bipartite subgraph induced by N_d(c) and internal checks
//   F[c]        = data-node set covering EVERY cycle in S_d(c) (so the
//                 residual is acyclic and inner min-sum BP is exact).
//                 Seeded with N(c) so outgoing-edge marginalisation is
//                 trivial; then extended by iterative max-degree
//                 cycle-removal until no cycle remains in S_d(c)\F.
//   R[c]        = N_d(c) \ F[c]    -- the residual
//
// Per outer iteration, per check c:
//   for each x_F in {0,1}^|F[c]|:                                   2^|F|
//     skip if any F-only internal-check syndrome is violated
//     run exact min-sum BP on the residual subgraph S_d(c) \ F,    O(|S_d|)
//       with effective syndromes s'_{c'} = s_{c'} ⊕ ⊕_{v ∈ F ∩ N(c')} x_v
//     take hard decisions of the residual, verify feasibility,
//     compute residual cost
//     marginalise total cost into per-outgoing-edge W_0, W_1
//   m_{c→v} = W_1 − W_0  for each outgoing edge
//
// Because F is a complete FVS of S_d(c), the residual is a tree (forest)
// and inner min-sum BP converges to the exact min-cost residual config
// in O(diameter) sweeps.
// ---------------------------------------------------------------------------
template<typename LoggerT>
class FvsDecoder : public Decoder {
public:
    static constexpr int MAX_FVS         = 30;
    static constexpr int INNER_BP_ITERS  = 16;  // generous; trees converge faster

    explicit FvsDecoder(const FlashBPBase& bp, LoggerT logger, int degree);

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

    // ── Per-check neighbourhoods ────────────────────────────────────────────
    std::vector<std::vector<int>> nbhd_data_;     // sorted data-node indices in N_d(c)
    std::vector<std::vector<int>> fvs_;           // F[c] subset of N_d(c)
    std::vector<std::vector<int>> residual_;      // R[c] = N_d(c) \ F[c]

    // ── F-only internal-check constraints (all of c''s vars are in F) ───────
    // Checked once per x_F before running inner BP; an F-only violation kills
    // the x_F branch immediately.
    struct FOnlyCheck { uint32_t f_mask; int check_idx; };
    std::vector<std::vector<FOnlyCheck>> f_only_checks_;

    // ── Residual bipartite subgraph for inner tree min-sum BP ───────────────
    // For each check c, the structure of the graph that remains once F[c] is
    // removed.  Edges are labelled 0..n_edges-1 in the order they're added.
    struct ResidualGraph {
        std::vector<int>              check_indices;  // [rc] -> detector index
        std::vector<uint32_t>         check_f_masks;  // [rc] -> mask over F (for eff_syn)
        std::vector<uint32_t>         check_r_masks;  // [rc] -> mask over R (for feasibility)
        std::vector<std::vector<int>> check_edges;    // [rc] -> list of edge indices
        std::vector<std::vector<int>> axis_edges;     // [k]  -> list of edge indices
        int                           n_edges = 0;
    };
    std::vector<ResidualGraph> residual_graph_;

    // ── Per-axis incoming-edge map (-1 = deep, use channel prior) ───────────
    std::vector<std::vector<int>> fvs_axis_edge_;

    // For each Tanner edge ei: position of edges_[ei].var inside fvs_[edges_[ei].check]
    // (always defined since F[c] ⊇ N(c) by construction).
    std::vector<int> edge_fvs_pos_;

    LoggerT      logger_;
    unsigned int shot_counter_ = 0;
};

extern template class FvsDecoder<Logger<false>>;
extern template class FvsDecoder<Logger<true>>;
extern template class FvsDecoder<DecodeLogger<true>>;
extern template class FvsDecoder<RecordLogger>;
extern template class FvsDecoder<TensorLogger>;
extern template class FvsDecoder<MLLogger>;
