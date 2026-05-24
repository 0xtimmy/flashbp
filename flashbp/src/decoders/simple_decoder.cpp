#include "simple_decoder.h"
#include "../flashbp_core.h"
#include <cmath>
#include <limits>
#include <stdexcept>

// ── Constructor ───────────────────────────────────────────────────────────────

template<typename LoggerT>
SimpleDecoder<LoggerT>::SimpleDecoder(const FlashBPBase& bp, LoggerT logger)
    : num_detectors_(bp.num_detectors)
    , num_errors_(bp.num_errors)
    , logger_(std::move(logger))
{
    const auto& H           = bp.H_raw();
    const auto& error_probs = bp.error_probs_raw();

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

    logger_("SimpleDecoder constructed  edges=" + std::to_string(edges_.size()), 1);
}

// ── operator() ────────────────────────────────────────────────────────────────

template<typename LoggerT>
std::vector<uint8_t> SimpleDecoder<LoggerT>::operator()(
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
            std::to_string([&]{ int w=0; for (auto b:syndrome) w+=b; return w; }()), 3);

    int num_edges = (int)edges_.size();

    std::vector<double> msg_v2c(num_edges);
    std::vector<double> msg_c2v(num_edges, 0.0);
    for (int i = 0; i < num_edges; ++i)
        msg_v2c[i] = ch_llr_[edges_[i].var];

    std::vector<uint8_t> decision(num_errors_);

    for (int iter = 0; iter < max_iter; ++iter) {
        if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>)
            logger_.set_iteration(iter);
        logger_("iteration=" + std::to_string(iter), 3);

        // ---- check node update (min-sum) ------------------------------------
        for (int d = 0; d < num_detectors_; ++d) {
            const auto& eidx = check_edges_[d];
            int deg = (int)eidx.size();
            if (deg == 0) continue;

            int sign_prod = (syndrome[d] == 1) ? -1 : 1;
            for (int i : eidx)
                sign_prod *= (msg_v2c[i] >= 0.0) ? 1 : -1;

            double min1 = std::numeric_limits<double>::infinity();
            double min2 = std::numeric_limits<double>::infinity();
            int    argmin1 = 0;
            for (int k = 0; k < deg; ++k) {
                double m = std::abs(msg_v2c[eidx[k]]);
                if (m < min1) { min2 = min1; min1 = m; argmin1 = k; }
                else if (m < min2) { min2 = m; }
            }

            for (int k = 0; k < deg; ++k) {
                int    ei       = eidx[k];
                int    v_sign   = (msg_v2c[ei] >= 0.0) ? 1 : -1;
                int    out_sign = sign_prod * v_sign;
                double out_mag  = (k == argmin1) ? min2 : min1;
                msg_c2v[ei] = (double)out_sign * out_mag;
            }
        }

        // ---- variable node update + hard decision ---------------------------
        for (int e = 0; e < num_errors_; ++e) {
            double total = ch_llr_[e];
            for (int i : var_edges_[e]) total += msg_c2v[i];

            decision[e] = (total < 0.0) ? 1u : 0u;

            for (int i : var_edges_[e])
                msg_v2c[i] = total - msg_c2v[i];
        }

        if constexpr (std::is_base_of_v<RecordLogger, LoggerT>)
            logger_.record_iteration(iter, syndrome, decision, msg_v2c, msg_c2v);

        // ---- convergence check ---------------------------------------------
        bool converged = true;
        for (int d = 0; d < num_detectors_ && converged; ++d) {
            int parity = 0;
            for (int ei : check_edges_[d])
                parity ^= decision[edges_[ei].var];
            if (parity != (int)syndrome[d]) converged = false;
        }

        if (converged) {
            logger_("converged at iter=" + std::to_string(iter), 3);
            set_decode_stats(true, iter + 1);
            return decision;
        }
    }

    logger_("max_iter=" + std::to_string(max_iter) + " reached without convergence", 2);
    set_decode_stats(false, max_iter);
    return decision;
}

// ── Explicit instantiations ───────────────────────────────────────────────────

template class SimpleDecoder<Logger<false>>;
template class SimpleDecoder<Logger<true>>;
template class SimpleDecoder<DecodeLogger<true>>;
template class SimpleDecoder<RecordLogger>;
template class SimpleDecoder<TensorLogger>;
template class SimpleDecoder<MLLogger>;
