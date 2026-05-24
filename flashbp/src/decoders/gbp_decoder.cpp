#include "gbp_decoder.h"

#include "../flashbp_core.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace {

bool gbp_region_is_active(const GBPRegion& region,
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

std::vector<int> active_region_indices(const std::vector<GBPRegion>& regions,
                                       const std::vector<uint8_t>& syndrome) {
    std::vector<int> active;
    for (int i = 0; i < (int)regions.size(); ++i) {
        if (gbp_region_is_active(regions[i], syndrome))
            active.push_back(i);
    }
    return active;
}

} // anonymous namespace

template<typename LoggerT>
GBPDecoder<LoggerT>::GBPDecoder(const FlashBPBase& bp,
                                LoggerT logger,
                                int degree,
                                std::unique_ptr<RegionGroupingPolicy> policy,
                                std::unique_ptr<GBPBackend> backend,
                                double oscillation_boost,
                                double oscillation_boost_cap)
    : num_detectors_(bp.num_detectors)
    , num_errors_(bp.num_errors)
    , degree_(degree)
    , oscillation_boost_(oscillation_boost)
    , oscillation_boost_cap_(oscillation_boost_cap)
    , policy_(std::move(policy))
    , backend_(std::move(backend))
    , logger_(std::move(logger))
{
    if (!policy_)
        throw std::invalid_argument("GBPDecoder: region grouping policy is null.");
    if (!backend_)
        throw std::invalid_argument("GBPDecoder: backend is null.");
    if (oscillation_boost_ < 1.0)
        throw std::invalid_argument("GBPDecoder: oscillation_boost must be >= 1.");
    if (oscillation_boost_cap_ < 1.0)
        throw std::invalid_argument("GBPDecoder: oscillation_boost_cap must be >= 1.");

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
    backend_->prepare(regions_, ch_llr_);

    int max_axes = 0;
    int max_internal = 0;
    for (const auto& region : regions_) {
        max_axes = std::max(max_axes, (int)region.data.size());
        max_internal = std::max(max_internal, (int)region.internal_checks.size());
    }

    logger_("GBPDecoder constructed  policy=" + std::string(policy_->name()) +
            "  backend=" + std::string(backend_->name()) +
            "  degree=" + std::to_string(degree_) +
            "  regions=" + std::to_string(regions_.size()) +
            "  edges=" + std::to_string(edges_.size()) +
            "  max_axes=" + std::to_string(max_axes) +
            "  max_internal_checks=" + std::to_string(max_internal) +
            "  oscillation_boost=" + std::to_string(oscillation_boost_) +
            "  oscillation_boost_cap=" + std::to_string(oscillation_boost_cap_),
            1);

    if constexpr (std::is_same_v<LoggerT, GBPLogger>) {
        logger_.record_gbp_start(
            policy_->name(),
            backend_->name(),
            degree_,
            num_detectors_,
            num_errors_,
            (int)edges_.size(),
            regions_);
    }
}

template<typename LoggerT>
std::vector<uint8_t> GBPDecoder<LoggerT>::operator()(
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
    std::unordered_map<std::string, int> seen_decisions;
    double region_message_scale = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>)
            logger_.set_iteration((unsigned int)iter);
        logger_("GBP iteration=" + std::to_string(iter), 3);

        std::fill(next_c2v.begin(), next_c2v.end(), 0.0);
        std::fill(next_c2v_count.begin(), next_c2v_count.end(), 0);

        backend_->update_regions(syndrome, msg_v2c, next_c2v, next_c2v_count);

        for (int ei = 0; ei < num_edges; ++ei)
            msg_c2v[ei] = next_c2v_count[ei]
                         ? region_message_scale *
                               (next_c2v[ei] / (double)next_c2v_count[ei])
                         : 0.0;

        for (int e = 0; e < num_errors_; ++e) {
            double total = ch_llr_[e];
            for (int i : var_edges_[e]) total += msg_c2v[i];
            decision[e] = (total < 0.0) ? 1u : 0u;
            for (int i : var_edges_[e])
                msg_v2c[i] = total - msg_c2v[i];
        }

        if constexpr (std::is_same_v<LoggerT, GBPLogger>) {
            logger_.record_gbp_iteration(
                iter,
                syndrome,
                decision,
                msg_v2c,
                msg_c2v,
                active_region_indices(regions_, syndrome));
        } else if constexpr (std::is_base_of_v<RecordLogger, LoggerT>) {
            logger_.record_iteration(iter, syndrome, decision, msg_v2c, msg_c2v);
        }

        bool converged = true;
        for (int d = 0; d < num_detectors_ && converged; ++d) {
            int parity = 0;
            for (int ei : check_edges_[d])
                parity ^= decision[edges_[ei].var];
            if (parity != (int)syndrome[d]) converged = false;
        }

        if (converged) {
            logger_("GBP converged at iter=" + std::to_string(iter), 3);
            set_decode_stats(true, iter + 1);
            return decision;
        }

        if (oscillation_boost_ > 1.0) {
            const std::string key(
                reinterpret_cast<const char*>(decision.data()),
                decision.size());
            int& seen = seen_decisions[key];
            if (seen > 0) {
                const double old_scale = region_message_scale;
                region_message_scale = std::min(
                    oscillation_boost_cap_,
                    region_message_scale * oscillation_boost_);
                if (region_message_scale > old_scale) {
                    logger_(
                        "GBP oscillation detected  iter=" +
                        std::to_string(iter) +
                        "  repeat_count=" + std::to_string(seen) +
                        "  region_message_scale=" +
                        std::to_string(region_message_scale),
                        2);
                }
            }
            ++seen;
        }
    }

    logger_("GBP max_iter=" + std::to_string(max_iter) +
            " reached without convergence", 2);
    set_decode_stats(false, max_iter);
    return decision;
}

template class GBPDecoder<Logger<false>>;
template class GBPDecoder<Logger<true>>;
template class GBPDecoder<DecodeLogger<true>>;
template class GBPDecoder<RecordLogger>;
template class GBPDecoder<TensorLogger>;
template class GBPDecoder<GBPLogger>;
template class GBPDecoder<MLLogger>;
