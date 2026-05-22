#include "gbp_decoder.h"

#include "../flashbp_core.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

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

bool region_is_active(const GBPRegion& region,
                      const std::vector<uint8_t>& syndrome) {
    switch (region.activation) {
    case GBPRegionActivation::Always:
        return true;
    case GBPRegionActivation::AnyCheckActive:
        for (int c : region.cycle_checks)
            if (syndrome[c] & 1) return true;
        return false;
    case GBPRegionActivation::AllChecksActive:
        if (region.cycle_checks.empty()) return false;
        for (int c : region.cycle_checks)
            if ((syndrome[c] & 1) == 0) return false;
        return true;
    }
    return true;
}

} // anonymous namespace

template<typename LoggerT>
GBPDecoder<LoggerT>::GBPDecoder(const FlashBPBase& bp,
                                LoggerT logger,
                                int degree,
                                std::unique_ptr<RegionGroupingPolicy> policy)
    : num_detectors_(bp.num_detectors)
    , num_errors_(bp.num_errors)
    , degree_(degree)
    , policy_(std::move(policy))
    , logger_(std::move(logger))
{
    if (!policy_)
        throw std::invalid_argument("GBPDecoder: region grouping policy is null.");

    const auto& H = bp.H_raw();
    const auto& error_probs = bp.error_probs_raw();

    var_edges_.assign(num_errors_, {});
    check_edges_.assign(num_detectors_, {});

    for (int d = 0; d < num_detectors_; ++d) {
        for (int e = 0; e < num_errors_; ++e) {
            if (H[(size_t)d * num_errors_ + e]) {
                int idx = (int)edges_.size();
                edges_.push_back({d, e});
                var_edges_[e].push_back(idx);
                check_edges_[d].push_back(idx);
            }
        }
    }

    ch_llr_.resize(num_errors_);
    for (int e = 0; e < num_errors_; ++e) {
        double p = std::max(1e-12, std::min(1.0 - 1e-12, error_probs[e]));
        ch_llr_[e] = std::log((1.0 - p) / p);
    }

    regions_ = policy_->build_regions(
        num_detectors_, num_errors_, edges_, var_edges_, check_edges_,
        edge_axis_pos_);

    int max_axes = 0;
    int max_internal = 0;
    for (const auto& region : regions_) {
        max_axes = std::max(max_axes, (int)region.data.size());
        max_internal = std::max(max_internal, (int)region.internal_checks.size());
    }

    logger_("GBPDecoder constructed  policy=" + std::string(policy_->name()) +
            "  degree=" + std::to_string(degree_) +
            "  regions=" + std::to_string(regions_.size()) +
            "  edges=" + std::to_string(edges_.size()) +
            "  max_axes=" + std::to_string(max_axes) +
            "  max_internal_checks=" + std::to_string(max_internal), 1);
}

template<typename LoggerT>
std::vector<uint8_t> GBPDecoder<LoggerT>::operator()(
    const std::vector<uint8_t>& syndrome,
    int max_iter)
{
    if ((int)syndrome.size() != num_detectors_)
        throw std::invalid_argument("Syndrome length must equal num_detectors.");

    if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>) {
        logger_.set_shot(shot_counter_);
        logger_.set_iteration(0);
    }
    ++shot_counter_;

    logger_("GBP decode called  syndrome_weight=" +
            std::to_string([&]{ int w = 0; for (auto b : syndrome) w += b; return w; }()),
            3);

    const int num_edges = (int)edges_.size();
    std::vector<double> msg_v2c(num_edges);
    std::vector<double> msg_c2v(num_edges, 0.0);
    std::vector<double> next_c2v(num_edges, 0.0);
    std::vector<int>    next_c2v_count(num_edges, 0);
    for (int i = 0; i < num_edges; ++i)
        msg_v2c[i] = ch_llr_[edges_[i].var];

    std::vector<uint8_t> decision(num_errors_);

    size_t max_states = 1;
    for (const auto& region : regions_)
        max_states = std::max(max_states, (size_t)1 << region.data.size());
    std::vector<double> weight(max_states);
    std::vector<uint8_t> bad(max_states);

    constexpr double INF = std::numeric_limits<double>::infinity();
    constexpr double SAT = 1e30;

    for (int iter = 0; iter < max_iter; ++iter) {
        if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>)
            logger_.set_iteration((unsigned int)iter);
        logger_("GBP iteration=" + std::to_string(iter), 3);

        std::fill(next_c2v.begin(), next_c2v.end(), 0.0);
        std::fill(next_c2v_count.begin(), next_c2v_count.end(), 0);

        for (const GBPRegion& region : regions_) {
            if (!region_is_active(region, syndrome))
                continue;
            const int K = (int)region.data.size();
            if (K == 0) continue;
            const size_t N = (size_t)1 << K;

            std::vector<double> incoming(K);
            for (int k = 0; k < K; ++k) {
                int ei = region.axis_edge[k];
                incoming[k] = (ei >= 0) ? msg_v2c[ei] : ch_llr_[region.data[k]];
            }

            for (size_t idx = 0; idx < N; ++idx) {
                double w = 0.0;
                for (int k = 0; k < K; ++k) {
                    int x = (int)((idx >> k) & 1);
                    w += minsum_cost(x, incoming[k]);
                }
                weight[idx] = w;
            }

            for (size_t idx = 0; idx < N; ++idx) {
                uint8_t violated = 0;
                for (const auto& ic : region.internal_checks) {
                    int s = syndrome[ic.check_idx] & 1;
                    int sum = popcount32((uint32_t)(idx & ic.mask)) & 1;
                    if ((sum ^ s) != 0) {
                        violated = 1;
                        break;
                    }
                }
                bad[idx] = violated;
            }

            for (const auto& out : region.outputs) {
                int ei = out.edge_idx;
                int k = out.axis;
                double l_v = msg_v2c[ei];

                double W0 = INF;
                double W1 = INF;
                for (size_t idx = 0; idx < N; ++idx) {
                    if (bad[idx]) continue;
                    int x = (int)((idx >> k) & 1);
                    double w_ext = weight[idx] - minsum_cost(x, l_v);
                    if (x == 0) W0 = std::min(W0, w_ext);
                    else        W1 = std::min(W1, w_ext);
                }

                double out_msg = 0.0;
                if      (W0 == INF && W1 == INF) out_msg = 0.0;
                else if (W0 == INF)              out_msg = -SAT;
                else if (W1 == INF)              out_msg = +SAT;
                else                             out_msg = W1 - W0;

                next_c2v[ei] += out_msg;
                next_c2v_count[ei] += 1;
            }
        }

        for (int ei = 0; ei < num_edges; ++ei)
            msg_c2v[ei] = next_c2v_count[ei]
                         ? next_c2v[ei] / (double)next_c2v_count[ei]
                         : 0.0;

        for (int e = 0; e < num_errors_; ++e) {
            double total = ch_llr_[e];
            for (int i : var_edges_[e]) total += msg_c2v[i];
            decision[e] = (total < 0.0) ? 1u : 0u;
            for (int i : var_edges_[e])
                msg_v2c[i] = total - msg_c2v[i];
        }

        if constexpr (std::is_base_of_v<RecordLogger, LoggerT>)
            logger_.record_iteration(iter, syndrome, decision, msg_v2c, msg_c2v);

        bool converged = true;
        for (int d = 0; d < num_detectors_ && converged; ++d) {
            int parity = 0;
            for (int ei : check_edges_[d])
                parity ^= decision[edges_[ei].var];
            if (parity != (int)syndrome[d]) converged = false;
        }

        if (converged) {
            logger_("GBP converged at iter=" + std::to_string(iter), 3);
            return decision;
        }
    }

    logger_("GBP max_iter=" + std::to_string(max_iter) +
            " reached without convergence", 2);
    return decision;
}

template class GBPDecoder<Logger<false>>;
template class GBPDecoder<Logger<true>>;
template class GBPDecoder<DecodeLogger<true>>;
template class GBPDecoder<RecordLogger>;
template class GBPDecoder<TensorLogger>;
template class GBPDecoder<MLLogger>;
