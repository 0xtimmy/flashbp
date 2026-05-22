#include "ml_decoder.h"
#include "../flashbp_core.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <type_traits>

#ifdef FLASHBP_HAS_TORCH
  #include <torch/torch.h>
#endif

namespace {

// Numerically stable log(exp(a) + exp(b)) handling -inf cleanly.
inline double logsumexp(double a, double b) {
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    const double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

inline uint64_t pack_bits(const std::vector<uint8_t>& bits) {
    uint64_t out = 0;
    for (int i = 0; i < (int)bits.size(); ++i)
        if (bits[i] & 1u) out |= (uint64_t{1} << i);
    return out;
}

constexpr int64_t ML_RECORD_MAX_STATES = 4096;

struct BranchDivergence {
    double kl_0_to_1 = 0.0;
    double kl_1_to_0 = 0.0;
    double js = 0.0;
};

double logsumexp_all(const std::vector<double>& values) {
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    double m = NEG_INF;
    for (double v : values)
        if (v > m) m = v;
    if (m == NEG_INF) return NEG_INF;
    double sum = 0.0;
    for (double v : values)
        if (v != NEG_INF) sum += std::exp(v - m);
    return m + std::log(sum);
}

BranchDivergence branch_divergence_from_dist(
    const std::vector<double>& dist,
    int64_t                    flip)
{
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    const double logz = logsumexp_all(dist);
    if (logz == NEG_INF)
        return {};

    BranchDivergence out;
    bool inf_01 = false;
    bool inf_10 = false;
    const double log_half = std::log(0.5);

    for (int64_t state = 0; state < (int64_t)dist.size(); ++state) {
        const double lp = dist[(size_t)state] - logz;
        const int64_t other = state ^ flip;
        const double lq = (0 <= other && other < (int64_t)dist.size())
            ? dist[(size_t)other] - logz
            : NEG_INF;

        const bool p_fin = std::isfinite(lp);
        const bool q_fin = std::isfinite(lq);
        if (p_fin) {
            const double p = std::exp(lp);
            if (q_fin) out.kl_0_to_1 += p * (lp - lq);
            else       inf_01 = true;
            const double lm = logsumexp(lp, lq) + log_half;
            out.js += 0.5 * p * (lp - lm);
        }
        if (q_fin) {
            const double q = std::exp(lq);
            if (p_fin) out.kl_1_to_0 += q * (lq - lp);
            else       inf_10 = true;
            const double lm = logsumexp(lp, lq) + log_half;
            out.js += 0.5 * q * (lq - lm);
        }
    }

    if (inf_01) out.kl_0_to_1 = std::numeric_limits<double>::infinity();
    if (inf_10) out.kl_1_to_0 = std::numeric_limits<double>::infinity();
    return out;
}

std::vector<double> class_slice_from_dist(
    const std::vector<double>& dist,
    int                        num_detectors,
    int                        num_observables,
    uint64_t                   syndrome_state)
{
    const int n_classes = 1 << num_observables;
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    std::vector<double> out((size_t)n_classes, NEG_INF);
    for (int cls = 0; cls < n_classes; ++cls) {
        const uint64_t state =
            syndrome_state | ((uint64_t)cls << num_detectors);
        if (state < dist.size())
            out[(size_t)cls] = dist[(size_t)state];
    }
    return out;
}

std::vector<double> initial_class_slice(
    int      num_observables,
    uint64_t syndrome_state)
{
    const int n_classes = 1 << num_observables;
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    std::vector<double> out((size_t)n_classes, NEG_INF);
    if (syndrome_state == 0)
        out[0] = 0.0;
    return out;
}

std::vector<double> xor_convolve_classes(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    std::vector<double> out(a.size(), NEG_INF);
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == NEG_INF) continue;
        for (size_t j = 0; j < b.size(); ++j) {
            if (b[j] == NEG_INF) continue;
            out[i ^ j] = logsumexp(out[i ^ j], a[i] + b[j]);
        }
    }
    return out;
}

std::vector<int> order_errors_by_syndrome(
    const std::vector<uint8_t>& H,
    int                         num_detectors,
    int                         num_errors,
    const std::vector<uint8_t>& syndrome,
    const std::vector<int>&     errors)
{
    std::vector<int> sources;
    for (int d = 0; d < num_detectors; ++d)
        if (syndrome[d] & 1u) sources.push_back(d);

    if (sources.empty()) {
        std::vector<int> out = errors;
        std::sort(out.begin(), out.end());
        return out;
    }

    const int total = num_errors + num_detectors;
    std::vector<int> dist(total, std::numeric_limits<int>::max());
    std::queue<int> q;
    for (int d : sources) {
        const int node = num_errors + d;
        dist[node] = 0;
        q.push(node);
    }

    while (!q.empty()) {
        const int u = q.front();
        q.pop();
        if (u < num_errors) {
            const int e = u;
            for (int d = 0; d < num_detectors; ++d) {
                if (!H[(size_t)d * num_errors + e]) continue;
                const int v = num_errors + d;
                if (dist[v] <= dist[u] + 1) continue;
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        } else {
            const int d = u - num_errors;
            for (int e = 0; e < num_errors; ++e) {
                if (!H[(size_t)d * num_errors + e]) continue;
                if (dist[e] <= dist[u] + 1) continue;
                dist[e] = dist[u] + 1;
                q.push(e);
            }
        }
    }

    std::vector<int> out = errors;
    std::sort(out.begin(), out.end(), [&](int a, int b) {
        if (dist[a] != dist[b]) return dist[a] < dist[b];
        return a < b;
    });
    return out;
}

} // anonymous namespace

// ── Constructor ───────────────────────────────────────────────────────────────

template<typename LoggerT>
MaximumLikelihoodDecoder<LoggerT>::MaximumLikelihoodDecoder(
    const FlashBPBase& bp,
    LoggerT            logger,
    int                bond_dim)
    : num_detectors_(bp.num_detectors)
    , num_errors_(bp.num_errors)
    , num_observables_(bp.num_observables)
    , bond_dim_(bond_dim)
    , H_(bp.H_raw())
    , L_(bp.L_raw())
    , logger_(std::move(logger))
{
    if (num_observables_ > 30)
        throw std::invalid_argument(
            "MaximumLikelihoodDecoder: " + std::to_string(num_observables_) +
            " observables — class enumeration needs 2^k buckets and k must "
            "fit in an int. Consider per-block decoding.");

    const auto& probs = bp.error_probs_raw();
    log_p_  .resize(num_errors_);
    log_1mp_.resize(num_errors_);
    for (int e = 0; e < num_errors_; ++e) {
        const double p = std::max(1e-300, std::min(1.0 - 1e-300, probs[e]));
        log_p_[e]   = std::log(p);
        log_1mp_[e] = std::log(1.0 - p);
    }

    logger_("MaximumLikelihoodDecoder constructed  errors=" +
            std::to_string(num_errors_) +
            "  observables=" + std::to_string(num_observables_) +
            "  bond_dim="    + std::to_string(bond_dim_), 1);
}

// ── class_log_probs ──────────────────────────────────────────────────────────
//
// Brute-force enumeration scaffold.  Replace with a tensor-network contraction
// once the TN structure is built; the rest of operator() consumes only the
// return value of this function, so swapping is a localised change.

template<typename LoggerT>
std::vector<double> MaximumLikelihoodDecoder<LoggerT>::class_log_probs(
    const std::vector<uint8_t>& syndrome) const
{
    std::vector<double> split_logp;
    if (class_log_probs_split(syndrome, split_logp))
        return split_logp;

#ifdef FLASHBP_HAS_TORCH
    return class_log_probs_torch(syndrome);
#else
    {
        const int state_bits = num_detectors_ + num_observables_;
        if (state_bits >= 63)
            throw std::runtime_error(
                "MaximumLikelihoodDecoder: CPU dense contraction needs "
                "num_detectors + num_observables < 63.");

        constexpr int MAX_DENSE_STATE_BITS = 30;
        if (state_bits > MAX_DENSE_STATE_BITS)
            throw std::runtime_error(
                "MaximumLikelihoodDecoder: CPU dense contraction would allocate "
                "2^" + std::to_string(state_bits) + " states. Rebuild with "
                "LibTorch for the Torch backend or use the forthcoming compressed "
                "TN contraction path.");

        const int n_classes = 1 << num_observables_;
        const int64_t n_states = int64_t{1} << state_bits;
        constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

        std::vector<int64_t> flips(num_errors_, 0);
        for (int e = 0; e < num_errors_; ++e) {
            uint64_t flip = 0;
            for (int d = 0; d < num_detectors_; ++d)
                if (H_[(size_t)d * num_errors_ + e])
                    flip |= (uint64_t{1} << d);
            for (int o = 0; o < num_observables_; ++o)
                if (L_[(size_t)o * num_errors_ + e])
                    flip |= (uint64_t{1} << (num_detectors_ + o));
            flips[e] = (int64_t)flip;
        }

        std::vector<double> dist((size_t)n_states, NEG_INF);
        dist[0] = 0.0;

        if constexpr (std::is_base_of_v<MLLogger, LoggerT>) {
            logger_.record_ml_start(syndrome, num_detectors_, num_observables_, "cpu");
            const uint64_t syndrome_state = pack_bits(syndrome);
            logger_.record_ml_step(
                -1, -1, 0, state_bits, n_states,
                std::vector<int64_t>{0},
                std::vector<double>{0.0},
                initial_class_slice(num_observables_, syndrome_state));
        }

        std::vector<int> all_errors(num_errors_);
        std::iota(all_errors.begin(), all_errors.end(), 0);
        const std::vector<int> error_order =
            order_errors_by_syndrome(H_, num_detectors_, num_errors_,
                                     syndrome, all_errors);

        for (int order_pos = 0; order_pos < num_errors_; ++order_pos) {
            const int e = error_order[order_pos];
            const BranchDivergence branch_div =
                branch_divergence_from_dist(dist, flips[e]);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<double> next((size_t)n_states, NEG_INF);
            for (int64_t state = 0; state < n_states; ++state) {
                next[(size_t)state] = logsumexp(
                    next[(size_t)state],
                    dist[(size_t)state] + log_1mp_[e]);
                const int64_t state1 = state ^ flips[e];
                next[(size_t)state1] = logsumexp(
                    next[(size_t)state1],
                    dist[(size_t)state] + log_p_[e]);
            }
            dist = std::move(next);
            const auto t1 = std::chrono::steady_clock::now();
            const auto us =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            logger_("ML CPU contraction axis=" + std::to_string(e) +
                    "  order=" + std::to_string(order_pos) +
                    "  duration_us=" + std::to_string(us) +
                    "  js=" + std::to_string(branch_div.js) +
                    "  states=" + std::to_string(n_states), 2);

        if constexpr (std::is_base_of_v<MLLogger, LoggerT>) {
            const int64_t keep = std::min<int64_t>(ML_RECORD_MAX_STATES, n_states);
            std::vector<int64_t> order((size_t)n_states);
            for (int64_t state = 0; state < n_states; ++state)
                order[(size_t)state] = state;
            std::partial_sort(
                order.begin(), order.begin() + keep, order.end(),
                [&](int64_t a, int64_t b) {
                    return dist[(size_t)a] > dist[(size_t)b];
                });
            std::vector<int64_t> states_vec((size_t)keep);
            std::vector<double>  logp_vec((size_t)keep);
            for (int64_t i = 0; i < keep; ++i) {
                states_vec[(size_t)i] = order[(size_t)i];
                logp_vec[(size_t)i]   = dist[(size_t)order[(size_t)i]];
            }
            logger_.record_ml_step(
                order_pos, e, us, state_bits, n_states,
                std::move(states_vec), std::move(logp_vec),
                class_slice_from_dist(
                    dist, num_detectors_, num_observables_,
                    pack_bits(syndrome)),
                {},
                branch_div.kl_0_to_1,
                branch_div.kl_1_to_0,
                branch_div.js);
        }
        }

        std::vector<double> logp((size_t)n_classes, NEG_INF);
        const uint64_t syndrome_state = pack_bits(syndrome);
        for (int cls = 0; cls < n_classes; ++cls) {
            const uint64_t state =
                syndrome_state | ((uint64_t)cls << num_detectors_);
            logp[(size_t)cls] = dist[(size_t)state];
        }
        return logp;
    }

    if (num_errors_ > MAX_BRUTE_FORCE_BITS)
        throw std::runtime_error(
            "MaximumLikelihoodDecoder: brute-force scaffold limited to "
            "num_errors <= " + std::to_string(MAX_BRUTE_FORCE_BITS) +
            " (this code has " + std::to_string(num_errors_) +
            "). Use a smaller code, raise MAX_BRUTE_FORCE_BITS, or wait "
            "for the TN contraction backend.");

    const int    n_classes = 1 << num_observables_;
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    std::vector<double> logp(n_classes, NEG_INF);

    const size_t N = (size_t)1 << num_errors_;
    std::vector<uint8_t> err(num_errors_);

    for (size_t idx = 0; idx < N; ++idx) {
        // Decode idx -> binary error vector
        for (int e = 0; e < num_errors_; ++e)
            err[e] = (uint8_t)((idx >> e) & 1);

        // Syndrome check: H · err mod 2 == syndrome
        bool match = true;
        for (int d = 0; d < num_detectors_ && match; ++d) {
            int sum = 0;
            const uint8_t* row = &H_[(size_t)d * num_errors_];
            for (int e = 0; e < num_errors_; ++e)
                sum ^= row[e] & err[e];
            if (sum != (int)syndrome[d]) match = false;
        }
        if (!match) continue;

        // Logical class: L · err mod 2  (packed into an int)
        int logical = 0;
        for (int o = 0; o < num_observables_; ++o) {
            int sum = 0;
            const uint8_t* row = &L_[(size_t)o * num_errors_];
            for (int e = 0; e < num_errors_; ++e)
                sum ^= row[e] & err[e];
            logical |= (sum & 1) << o;
        }

        // Log-probability of this independent-error vector
        double lp = 0.0;
        for (int e = 0; e < num_errors_; ++e)
            lp += err[e] ? log_p_[e] : log_1mp_[e];

        logp[logical] = logsumexp(logp[logical], lp);
    }
    return logp;
#endif
}

template<typename LoggerT>
std::vector<double> MaximumLikelihoodDecoder<LoggerT>::class_log_probs_torch(
    const std::vector<uint8_t>& syndrome) const
{
#ifdef FLASHBP_HAS_TORCH
    const int state_bits = num_detectors_ + num_observables_;
    if (state_bits >= 63)
        throw std::runtime_error(
            "MaximumLikelihoodDecoder: dense Torch contraction needs "
            "num_detectors + num_observables < 63.");

    // Dense exact contraction over accumulated states:
    // state = syndrome_bits | (logical_bits << num_detectors).
    // This is GPU-friendly but still exponential in the state bit count, so it
    // is the exact contraction backend for small/medium DEMs, not yet the
    // large-code MPS/TN path.
    constexpr int MAX_DENSE_STATE_BITS = 30;
    if (state_bits > MAX_DENSE_STATE_BITS)
        throw std::runtime_error(
            "MaximumLikelihoodDecoder: dense Torch contraction would allocate "
            "2^" + std::to_string(state_bits) + " states. This needs the "
            "forthcoming MPS/TN contraction path; try a smaller code for the "
            "exact dense backend.");

    const int64_t n_states  = int64_t{1} << state_bits;
    const int64_t n_classes = int64_t{1} << num_observables_;
    const uint64_t syndrome_state = pack_bits(syndrome);
    auto device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA)
        : torch::Device(torch::kCPU);
    auto fopts = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto iopts = torch::TensorOptions().dtype(torch::kInt64).device(device);

    std::vector<int64_t> flips_host(num_errors_, 0);
    for (int e = 0; e < num_errors_; ++e) {
        uint64_t flip = 0;
        for (int d = 0; d < num_detectors_; ++d)
            if (H_[(size_t)d * num_errors_ + e])
                flip |= (uint64_t{1} << d);
        for (int o = 0; o < num_observables_; ++o)
            if (L_[(size_t)o * num_errors_ + e])
                flip |= (uint64_t{1} << (num_detectors_ + o));
        flips_host[e] = (int64_t)flip;
    }

    auto states = torch::arange(n_states, iopts);
    auto dist = torch::full({n_states},
                            -std::numeric_limits<double>::infinity(),
                            fopts);
    dist.index_put_({0}, 0.0);

    std::vector<int64_t> class_indices((size_t)n_classes);
    for (int64_t cls = 0; cls < n_classes; ++cls)
        class_indices[(size_t)cls] =
            (int64_t)(syndrome_state | ((uint64_t)cls << num_detectors_));

    auto class_idx = torch::from_blob(class_indices.data(), {n_classes},
                                      torch::TensorOptions().dtype(torch::kInt64))
                         .clone()
                         .to(device);

    auto torch_class_slice = [&]() {
        auto selected = dist.index_select(0, class_idx).to(torch::kCPU);
        std::vector<double> out((size_t)n_classes);
        auto acc = selected.accessor<double, 1>();
        for (int64_t i = 0; i < n_classes; ++i)
            out[(size_t)i] = acc[i];
        return out;
    };

    auto torch_branch_divergence = [&](const torch::Tensor& src_state) {
        BranchDivergence out;
        auto logz = torch::logsumexp(dist, {0});
        auto lp = dist - logz;
        auto lq = dist.index_select(0, src_state) - logz;
        auto finite_p = torch::isfinite(lp);
        auto finite_q = torch::isfinite(lq);
        auto both = torch::logical_and(finite_p, finite_q);
        auto zeros = torch::zeros_like(lp);

        const bool inf_01 = torch::any(
            torch::logical_and(finite_p, torch::logical_not(finite_q))
        ).item<bool>();
        const bool inf_10 = torch::any(
            torch::logical_and(finite_q, torch::logical_not(finite_p))
        ).item<bool>();

        if (inf_01) {
            out.kl_0_to_1 = std::numeric_limits<double>::infinity();
        } else {
            out.kl_0_to_1 = torch::where(
                both, torch::exp(lp) * (lp - lq), zeros
            ).sum().item<double>();
        }
        if (inf_10) {
            out.kl_1_to_0 = std::numeric_limits<double>::infinity();
        } else {
            out.kl_1_to_0 = torch::where(
                both, torch::exp(lq) * (lq - lp), zeros
            ).sum().item<double>();
        }

        auto lm = torch::logaddexp(lp, lq) + std::log(0.5);
        auto js0 = torch::where(
            finite_p, torch::exp(lp) * (lp - lm), zeros
        ).sum();
        auto js1 = torch::where(
            finite_q, torch::exp(lq) * (lq - lm), zeros
        ).sum();
        out.js = (0.5 * (js0 + js1)).item<double>();
        return out;
    };

    if constexpr (std::is_base_of_v<MLLogger, LoggerT>) {
        logger_.record_ml_start(
            syndrome,
            num_detectors_,
            num_observables_,
            device.type() == torch::kCUDA ? "cuda" : "cpu");
        logger_.record_ml_step(
            -1, -1, 0, state_bits, n_states,
            std::vector<int64_t>{0},
            std::vector<double>{0.0},
            initial_class_slice(num_observables_, syndrome_state));
    }

    std::vector<int> all_errors(num_errors_);
    std::iota(all_errors.begin(), all_errors.end(), 0);
    const std::vector<int> error_order =
        order_errors_by_syndrome(H_, num_detectors_, num_errors_,
                                 syndrome, all_errors);

    for (int order_pos = 0; order_pos < num_errors_; ++order_pos) {
        const int e = error_order[order_pos];
        const auto t0 = std::chrono::steady_clock::now();
        const double log0 = log_1mp_[e];
        const double log1 = log_p_[e];
        auto flip = torch::full({1}, flips_host[e], iopts);
        auto src_state = torch::bitwise_xor(states, flip);
        const BranchDivergence branch_div = torch_branch_divergence(src_state);
        auto no_error  = dist + log0;
        auto yes_error = dist.index_select(0, src_state) + log1;
        dist = torch::logaddexp(no_error, yes_error);
        if (device.type() == torch::kCUDA)
            torch::cuda::synchronize();
        const auto t1 = std::chrono::steady_clock::now();
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        logger_("ML Torch contraction axis=" + std::to_string(e) +
                "  order=" + std::to_string(order_pos) +
                "  duration_us=" + std::to_string(us) +
                "  js=" + std::to_string(branch_div.js) +
                "  states=" + std::to_string(n_states), 2);

        if constexpr (std::is_base_of_v<MLLogger, LoggerT>) {
            const int64_t keep = std::min<int64_t>(ML_RECORD_MAX_STATES, n_states);
            auto top = torch::topk(dist, keep);
            auto top_logp_cpu   = std::get<0>(top).to(torch::kCPU);
            auto top_states_cpu = std::get<1>(top).to(torch::kCPU);
            std::vector<int64_t> states_vec((size_t)keep);
            std::vector<double>  logp_vec((size_t)keep);
            auto p_acc = top_logp_cpu.accessor<double, 1>();
            auto top_s_acc = top_states_cpu.accessor<int64_t, 1>();
            for (int64_t i = 0; i < keep; ++i) {
                states_vec[(size_t)i] = top_s_acc[i];
                logp_vec[(size_t)i]   = p_acc[i];
            }
            logger_.record_ml_step(
                order_pos, e, us, state_bits, n_states,
                std::move(states_vec), std::move(logp_vec),
                torch_class_slice(),
                {},
                branch_div.kl_0_to_1,
                branch_div.kl_1_to_0,
                branch_div.js);
        }
    }

    auto selected = dist.index_select(0, class_idx).to(torch::kCPU);

    std::vector<double> out((size_t)n_classes);
    auto acc = selected.accessor<double, 1>();
    for (int64_t i = 0; i < n_classes; ++i)
        out[(size_t)i] = acc[i];

    logger_("ML Torch dense contraction  device=" +
            std::string(device.type() == torch::kCUDA ? "cuda" : "cpu") +
            "  state_bits=" + std::to_string(state_bits) +
            "  states=" + std::to_string(n_states), 2);
    return out;
#else
    (void)syndrome;
    throw std::runtime_error(
        "MaximumLikelihoodDecoder: LibTorch backend not linked. Rebuild with "
        "FLASHBP_ENABLE_TORCH=ON to use class_log_probs_torch().");
#endif
}

template<typename LoggerT>
bool MaximumLikelihoodDecoder<LoggerT>::class_log_probs_split(
    const std::vector<uint8_t>& syndrome,
    std::vector<double>&       out_logp) const
{
    if (num_observables_ <= 0) return false;

    const int total_nodes = num_errors_ + num_detectors_;
    std::vector<std::vector<int>> adj(total_nodes);
    for (int d = 0; d < num_detectors_; ++d) {
        for (int e = 0; e < num_errors_; ++e) {
            if (!H_[(size_t)d * num_errors_ + e]) continue;
            const int dn = num_errors_ + d;
            adj[e].push_back(dn);
            adj[dn].push_back(e);
        }
    }

    struct Component {
        std::vector<int> errors;
        std::vector<int> detectors;
        uint64_t logical_mask = 0;
    };

    std::vector<Component> comps;
    std::vector<uint8_t> seen(total_nodes, 0);
    for (int start = 0; start < total_nodes; ++start) {
        if (seen[start] || adj[start].empty()) continue;
        Component comp;
        std::queue<int> q;
        seen[start] = 1;
        q.push(start);
        while (!q.empty()) {
            const int u = q.front();
            q.pop();
            if (u < num_errors_) comp.errors.push_back(u);
            else                 comp.detectors.push_back(u - num_errors_);
            for (int v : adj[u]) {
                if (seen[v]) continue;
                seen[v] = 1;
                q.push(v);
            }
        }
        comps.push_back(std::move(comp));
    }

    if (comps.size() <= 1) return false;

    logger_("decoding X and Z separately  components=" +
            std::to_string(comps.size()), 1);

    const int n_classes = 1 << num_observables_;
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    out_logp.assign((size_t)n_classes, NEG_INF);
    out_logp[0] = 0.0;

    for (size_t ci = 0; ci < comps.size(); ++ci) {
        const auto& comp = comps[ci];
        const int cd = (int)comp.detectors.size();
        const int co = num_observables_;
        const int c_state_bits = cd + co;

        if (c_state_bits >= 63)
            throw std::runtime_error(
                "MaximumLikelihoodDecoder: split component state_bits >= 63.");

        const int64_t c_states = int64_t{1} << c_state_bits;
        std::vector<double> dist((size_t)c_states, NEG_INF);
        dist[0] = 0.0;

        std::vector<int64_t> flips(comp.errors.size(), 0);
        const std::vector<int> comp_error_order =
            order_errors_by_syndrome(H_, num_detectors_, num_errors_,
                                     syndrome, comp.errors);

        for (size_t local_e = 0; local_e < comp_error_order.size(); ++local_e) {
            const int e = comp_error_order[local_e];
            uint64_t flip = 0;
            for (int local_d = 0; local_d < cd; ++local_d) {
                const int d = comp.detectors[local_d];
                if (H_[(size_t)d * num_errors_ + e])
                    flip |= (uint64_t{1} << local_d);
            }
            for (int o = 0; o < num_observables_; ++o) {
                if (L_[(size_t)o * num_errors_ + e])
                    flip |= (uint64_t{1} << (cd + o));
            }
            flips[local_e] = (int64_t)flip;
        }

        if constexpr (std::is_base_of_v<MLLogger, LoggerT>) {
            if (ci == 0) {
                logger_.record_ml_start(
                    syndrome, num_detectors_, num_observables_, "split-cpu");
                uint64_t target_syndrome = 0;
                for (int local_d = 0; local_d < cd; ++local_d) {
                    const int d = comp.detectors[local_d];
                    if (syndrome[d] & 1u)
                        target_syndrome |= (uint64_t{1} << local_d);
                }
                logger_.record_ml_step(
                    -1, -1, 0, c_state_bits, c_states,
                    std::vector<int64_t>{0},
                    std::vector<double>{0.0},
                    xor_convolve_classes(
                        out_logp,
                        initial_class_slice(num_observables_, target_syndrome)));
            }
        }

        for (size_t local_e = 0; local_e < comp_error_order.size(); ++local_e) {
            const int e = comp_error_order[local_e];
            const BranchDivergence branch_div =
                branch_divergence_from_dist(dist, flips[local_e]);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<double> next((size_t)c_states, NEG_INF);
            for (int64_t state = 0; state < c_states; ++state) {
                next[(size_t)state] = logsumexp(
                    next[(size_t)state],
                    dist[(size_t)state] + log_1mp_[e]);
                const int64_t state1 = state ^ flips[local_e];
                next[(size_t)state1] = logsumexp(
                    next[(size_t)state1],
                    dist[(size_t)state] + log_p_[e]);
            }
            dist = std::move(next);
            const auto t1 = std::chrono::steady_clock::now();
            const auto us =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            logger_("ML split contraction component=" + std::to_string(ci) +
                    "  axis=" + std::to_string(e) +
                    "  order=" + std::to_string(local_e) +
                    "  duration_us=" + std::to_string(us) +
                    "  js=" + std::to_string(branch_div.js) +
                    "  states=" + std::to_string(c_states), 2);

            if constexpr (std::is_base_of_v<MLLogger, LoggerT>) {
                const int64_t keep = std::min<int64_t>(ML_RECORD_MAX_STATES, c_states);
                std::vector<int64_t> order((size_t)c_states);
                for (int64_t state = 0; state < c_states; ++state)
                    order[(size_t)state] = state;
                std::partial_sort(
                    order.begin(), order.begin() + keep, order.end(),
                    [&](int64_t a, int64_t b) {
                        return dist[(size_t)a] > dist[(size_t)b];
                    });
                std::vector<int64_t> states_vec((size_t)keep);
                std::vector<double>  logp_vec((size_t)keep);
                for (int64_t i = 0; i < keep; ++i) {
                    const int64_t local_state = order[(size_t)i];
                    uint64_t global_state = 0;
                    for (int local_d = 0; local_d < cd; ++local_d)
                        if ((local_state >> local_d) & 1)
                            global_state |= (uint64_t{1} << comp.detectors[local_d]);
                    for (int local_o = 0; local_o < co; ++local_o)
                        if ((local_state >> (cd + local_o)) & 1)
                            global_state |= (uint64_t{1}
                                             << (num_detectors_ + local_o));
                    states_vec[(size_t)i] = (int64_t)global_state;
                    logp_vec[(size_t)i]   = dist[(size_t)local_state];
                }
                uint64_t target_syndrome = 0;
                for (int local_d = 0; local_d < cd; ++local_d) {
                    const int d = comp.detectors[local_d];
                    if (syndrome[d] & 1u)
                        target_syndrome |= (uint64_t{1} << local_d);
                }
                auto partial_comp_logp = class_slice_from_dist(
                    dist, cd, num_observables_, target_syndrome);
                logger_.record_ml_step(
                    (int)local_e, e, us, c_state_bits, c_states,
                    std::move(states_vec), std::move(logp_vec),
                    xor_convolve_classes(out_logp, partial_comp_logp),
                    {},
                    branch_div.kl_0_to_1,
                    branch_div.kl_1_to_0,
                    branch_div.js);
            }
        }

        uint64_t target_syndrome = 0;
        for (int local_d = 0; local_d < cd; ++local_d) {
            const int d = comp.detectors[local_d];
            if (syndrome[d] & 1u)
                target_syndrome |= (uint64_t{1} << local_d);
        }

        std::vector<double> comp_logp((size_t)n_classes, NEG_INF);
        for (int local_cls = 0; local_cls < n_classes; ++local_cls) {
            const uint64_t local_state =
                target_syndrome | ((uint64_t)local_cls << cd);
            comp_logp[(size_t)local_cls] = dist[(size_t)local_state];
        }

        std::vector<double> combined((size_t)n_classes, NEG_INF);
        for (int a = 0; a < n_classes; ++a) {
            if (out_logp[(size_t)a] == NEG_INF) continue;
            for (int b = 0; b < n_classes; ++b) {
                if (comp_logp[(size_t)b] == NEG_INF) continue;
                const int cls = a ^ b;
                combined[(size_t)cls] = logsumexp(
                    combined[(size_t)cls],
                    out_logp[(size_t)a] + comp_logp[(size_t)b]);
            }
        }
        out_logp = std::move(combined);
    }

    return true;
}

// ── find_representative ──────────────────────────────────────────────────────
//
// Brute-force again — picks the highest-probability single error matching
// both the syndrome and the chosen logical class.  In the eventual TN
// implementation we'll likely run a quick approximate decoder (e.g. simple
// BP) to get a representative, then XOR with the right logical generator
// to land in the target class — no need to re-enumerate.

template<typename LoggerT>
std::vector<uint8_t> MaximumLikelihoodDecoder<LoggerT>::find_representative(
    const std::vector<uint8_t>& syndrome,
    int                         target_class) const
{
    const int rows = num_detectors_ + num_observables_;
    const int cols = num_errors_;

    // Augmented GF(2) matrix [H; L | rhs].
    std::vector<std::vector<uint8_t>> A(
        rows, std::vector<uint8_t>((size_t)cols + 1, 0));

    for (int d = 0; d < num_detectors_; ++d) {
        for (int e = 0; e < cols; ++e)
            A[d][e] = H_[(size_t)d * cols + e] & 1u;
        A[d][cols] = syndrome[d] & 1u;
    }
    for (int o = 0; o < num_observables_; ++o) {
        const int r = num_detectors_ + o;
        for (int e = 0; e < cols; ++e)
            A[r][e] = L_[(size_t)o * cols + e] & 1u;
        A[r][cols] = (uint8_t)((target_class >> o) & 1);
    }

    int pivot_row = 0;
    std::vector<int> pivot_cols;
    for (int col = 0; col < cols && pivot_row < rows; ++col) {
        int found = -1;
        for (int r = pivot_row; r < rows; ++r) {
            if (A[r][col]) { found = r; break; }
        }
        if (found < 0) continue;
        if (found != pivot_row)
            std::swap(A[found], A[pivot_row]);

        for (int r = 0; r < rows; ++r) {
            if (r == pivot_row || !A[r][col]) continue;
            for (int c = col; c <= cols; ++c)
                A[r][c] ^= A[pivot_row][c];
        }
        pivot_cols.push_back(col);
        ++pivot_row;
    }

    for (int r = pivot_row; r < rows; ++r) {
        bool all_zero = true;
        for (int c = 0; c < cols; ++c)
            if (A[r][c]) { all_zero = false; break; }
        if (all_zero && A[r][cols])
            throw std::runtime_error(
                "MaximumLikelihoodDecoder: no representative satisfies the "
                "chosen syndrome/logical class.");
    }

    std::vector<uint8_t> x(cols, 0);
    for (int r = 0; r < (int)pivot_cols.size(); ++r)
        x[pivot_cols[r]] = A[r][cols] & 1u;  // free variables remain zero
    return x;
}

// ── operator() ────────────────────────────────────────────────────────────────

template<typename LoggerT>
std::vector<uint8_t> MaximumLikelihoodDecoder<LoggerT>::operator()(
    const std::vector<uint8_t>& syndrome,
    int /*max_iter*/)
{
    if ((int)syndrome.size() != num_detectors_)
        throw std::invalid_argument("Syndrome length must equal num_detectors.");

    if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>) {
        logger_.set_shot(shot_counter_);
        logger_.set_iteration(0);
    }
    ++shot_counter_;

    logger_("decode called  syndrome_weight=" +
            std::to_string([&]{ int w=0; for (auto b:syndrome) w+=b; return w; }()),
            3);

    // 1. Compute log-probability of each logical class.
    //    (THIS is the step the tensor-network contraction replaces.)
    const std::vector<double> logp = class_log_probs(syndrome);

    // 2. argmax — most likely logical class.
    const int best_class =
        (int)std::distance(logp.begin(),
                           std::max_element(logp.begin(), logp.end()));

    if constexpr (std::is_base_of_v<MLLogger, LoggerT>)
        logger_.record_ml_final(logp, best_class);

    logger_("ML class chosen=" + std::to_string(best_class) +
            "  log_p="          + std::to_string(logp[best_class]), 2);

    // 3. Return any syndrome-consistent error in the chosen class.
    return find_representative(syndrome, best_class);
}

// ── Explicit instantiations ───────────────────────────────────────────────────
template class MaximumLikelihoodDecoder<Logger<false>>;
template class MaximumLikelihoodDecoder<Logger<true>>;
template class MaximumLikelihoodDecoder<DecodeLogger<true>>;
template class MaximumLikelihoodDecoder<RecordLogger>;
template class MaximumLikelihoodDecoder<TensorLogger>;
template class MaximumLikelihoodDecoder<MLLogger>;
template class MaximumLikelihoodDecoder<SurpriseMLLogger>;
