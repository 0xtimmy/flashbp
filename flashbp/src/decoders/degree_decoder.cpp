#include "degree_decoder.h"
#include "../flashbp_core.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace {

inline double minsum_cost(int x, double l) {
    return std::max(0.0, (x ? 1.0 : -1.0) * l);
}

inline int popcount32(uint32_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(v);
#else
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return (int)((((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
#endif
}

} // anonymous namespace

// ── Constructor ───────────────────────────────────────────────────────────────

template<typename LoggerT>
DegreeDecoder<LoggerT>::DegreeDecoder(const FlashBPBase& bp,
                                      LoggerT            logger,
                                      int                degree)
    : num_detectors_(bp.num_detectors)
    , num_errors_(bp.num_errors)
    , degree_(degree)
    , logger_(std::move(logger))
{
    if (degree < 1)
        throw std::invalid_argument("DegreeDecoder: degree must be >= 1.");

    const auto& H           = bp.H_raw();
    const auto& error_probs = bp.error_probs_raw();

    // ── Tanner edges + per-node adjacency ────────────────────────────────────
    var_edges_.assign(num_errors_,     {});
    check_edges_.assign(num_detectors_, {});

    for (int d = 0; d < num_detectors_; ++d)
        for (int e = 0; e < num_errors_; ++e)
            if (H[(size_t)d * num_errors_ + e]) {
                int idx = (int)edges_.size();
                edges_.push_back({d, e});
                var_edges_[e].push_back(idx);
                check_edges_[d].push_back(idx);
            }

    ch_llr_.resize(num_errors_);
    for (int e = 0; e < num_errors_; ++e) {
        double p = std::max(1e-12, std::min(1.0 - 1e-12, error_probs[e]));
        ch_llr_[e] = std::log((1.0 - p) / p);
    }

    // ── For each check c: N_d(c) via alternating BFS, then internal checks ──
    nbhd_data_      .assign(num_detectors_, {});
    axis_edge_      .assign(num_detectors_, {});
    internal_checks_.assign(num_detectors_, {});
    edge_axis_pos_  .assign(edges_.size(), -1);

    int max_K = 0;

    for (int c = 0; c < num_detectors_; ++c) {
        std::set<int>    data_in_nbhd;
        std::set<int>    visited_checks{c};
        std::vector<int> frontier_checks{c};

        for (int hop = 0; hop < degree; ++hop) {
            std::vector<int> new_data;
            for (int cc : frontier_checks)
                for (int ei : check_edges_[cc]) {
                    int v = edges_[ei].var;
                    if (data_in_nbhd.insert(v).second)
                        new_data.push_back(v);
                }
            if (hop + 1 == degree) break;

            std::vector<int> new_checks;
            for (int v : new_data)
                for (int ei : var_edges_[v]) {
                    int cc = edges_[ei].check;
                    if (visited_checks.insert(cc).second)
                        new_checks.push_back(cc);
                }
            if (new_checks.empty()) break;
            frontier_checks = std::move(new_checks);
        }

        nbhd_data_[c].assign(data_in_nbhd.begin(), data_in_nbhd.end());
        const int K = (int)nbhd_data_[c].size();
        max_K       = std::max(max_K, K);

        if (K > MAX_AXES)
            throw std::runtime_error(
                "DegreeDecoder: check " + std::to_string(c) +
                " has |N_d(c)|=" + std::to_string(K) +
                " > MAX_AXES=" + std::to_string(MAX_AXES) +
                " at degree=" + std::to_string(degree) + ".");

        std::vector<int> axis_of_v(num_errors_, -1);
        for (int k = 0; k < K; ++k)
            axis_of_v[nbhd_data_[c][k]] = k;

        axis_edge_[c].assign(K, -1);
        for (int ei : check_edges_[c]) {
            int v = edges_[ei].var;
            int k = axis_of_v[v];
            axis_edge_[c][k] = ei;
            edge_axis_pos_[ei] = k;
        }

        std::unordered_set<int> candidate_checks;
        for (int v : nbhd_data_[c])
            for (int ei : var_edges_[v])
                candidate_checks.insert(edges_[ei].check);

        for (int cc : candidate_checks) {
            uint32_t mask     = 0;
            bool     fully_in = true;
            for (int ei : check_edges_[cc]) {
                int k = axis_of_v[edges_[ei].var];
                if (k < 0) { fully_in = false; break; }
                mask |= (uint32_t)1 << k;
            }
            if (fully_in)
                internal_checks_[c].push_back({mask, cc});
        }
    }

    logger_("DegreeDecoder constructed  degree=" + std::to_string(degree_) +
            "  edges="    + std::to_string(edges_.size()) +
            "  max_axes=" + std::to_string(max_K), 2);
}

// ── operator() ────────────────────────────────────────────────────────────────

template<typename LoggerT>
std::vector<uint8_t> DegreeDecoder<LoggerT>::operator()(
    const std::vector<uint8_t>& syndrome,
    int max_iter)
{
    if ((int)syndrome.size() != num_detectors_)
        throw std::invalid_argument("Syndrome length must equal num_detectors.");
    reset_decode_stats();

    if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>) {
        logger_.set_shot(shot_counter_);
        logger_.set_iteration(0);
    }
    ++shot_counter_;

    logger_("decode called  syndrome_weight=" +
            std::to_string([&]{ int w=0; for (auto b:syndrome) w+=b; return w; }()),
            3);

    const int num_edges = (int)edges_.size();
    std::vector<double> msg_v2c(num_edges);
    std::vector<double> msg_c2v(num_edges, 0.0);
    for (int i = 0; i < num_edges; ++i)
        msg_v2c[i] = ch_llr_[edges_[i].var];

    std::vector<uint8_t> decision(num_errors_);

    // Pre-allocate the largest tensor we'll need (reused per check).
    size_t max_N = 1;
    for (auto& nb : nbhd_data_)
        max_N = std::max(max_N, (size_t)1 << nb.size());
    std::vector<double>  weight(max_N);
    std::vector<uint8_t> parity(max_N);

    constexpr double INF = std::numeric_limits<double>::infinity();
    constexpr double SAT = 1e30;

    for (int iter = 0; iter < max_iter; ++iter) {
        if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>)
            logger_.set_iteration((unsigned int)iter);
        logger_("iteration=" + std::to_string(iter), 3);

        // ── check-node update via degree-d region enumeration ─────────────
        for (int c = 0; c < num_detectors_; ++c) {
            const int K = (int)nbhd_data_[c].size();
            if (K == 0) continue;
            const size_t N = (size_t)1 << K;

            // Per-axis incoming LLR
            std::vector<double> l(K);
            for (int k = 0; k < K; ++k) {
                int ei = axis_edge_[c][k];
                l[k] = (ei >= 0) ? msg_v2c[ei] : ch_llr_[nbhd_data_[c][k]];
            }

            // Build weight tensor
            for (size_t idx = 0; idx < N; ++idx) {
                double w = 0.0;
                for (int k = 0; k < K; ++k) {
                    int x = (int)((idx >> k) & 1);
                    w += minsum_cost(x, l[k]);
                }
                weight[idx] = w;
            }

            // Build parity tensor (1 if any internal check is violated)
            for (size_t idx = 0; idx < N; ++idx) {
                uint8_t bad = 0;
                for (const auto& ic : internal_checks_[c]) {
                    int s   = syndrome[ic.check_idx] & 1;
                    int sum = popcount32((uint32_t)(idx & ic.mask)) & 1;
                    if ((sum ^ s) != 0) { bad = 1; break; }
                }
                parity[idx] = bad;
            }

            // Marginalise each outgoing edge to a signed LLR
            for (int ei : check_edges_[c]) {
                int    k   = edge_axis_pos_[ei];
                double l_v = msg_v2c[ei];

                double W0 = INF, W1 = INF;
                for (size_t idx = 0; idx < N; ++idx) {
                    if (parity[idx]) continue;
                    int    x     = (int)((idx >> k) & 1);
                    double w_ext = weight[idx] - minsum_cost(x, l_v);
                    if (x == 0) { if (w_ext < W0) W0 = w_ext; }
                    else        { if (w_ext < W1) W1 = w_ext; }
                }

                if      (W0 == INF && W1 == INF) msg_c2v[ei] =  0.0;
                else if (W0 == INF)              msg_c2v[ei] = -SAT;
                else if (W1 == INF)              msg_c2v[ei] = +SAT;
                else                             msg_c2v[ei] = W1 - W0;
            }
        }

        // ── variable-node update + hard decision ─────────────────────────
        for (int e = 0; e < num_errors_; ++e) {
            double total = ch_llr_[e];
            for (int i : var_edges_[e]) total += msg_c2v[i];
            decision[e] = (total < 0.0) ? 1u : 0u;
            for (int i : var_edges_[e])
                msg_v2c[i] = total - msg_c2v[i];
        }

        if constexpr (std::is_base_of_v<RecordLogger, LoggerT>)
            logger_.record_iteration(iter, syndrome, decision, msg_v2c, msg_c2v);

        // ── convergence check ────────────────────────────────────────────
        bool converged = true;
        for (int d = 0; d < num_detectors_ && converged; ++d) {
            int p = 0;
            for (int ei : check_edges_[d])
                p ^= decision[edges_[ei].var];
            if (p != (int)syndrome[d]) converged = false;
        }

        if (converged) {
            logger_("converged at iter=" + std::to_string(iter), 3);
            set_decode_stats(true, iter + 1);
            return decision;
        }
    }

    logger_("max_iter=" + std::to_string(max_iter) +
            " reached without convergence", 2);
    set_decode_stats(false, max_iter);
    return decision;
}

// ── Explicit instantiations ───────────────────────────────────────────────────
template class DegreeDecoder<Logger<false>>;
template class DegreeDecoder<Logger<true>>;
template class DegreeDecoder<DecodeLogger<true>>;
template class DegreeDecoder<RecordLogger>;
template class DegreeDecoder<TensorLogger>;
template class DegreeDecoder<MLLogger>;
