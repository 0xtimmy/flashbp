#include "surprise_ml_decoder.h"
#include "../flashbp_core.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace {

constexpr int64_t RECORD_MAX_STATES = 4096;

inline double logsumexp(double a, double b) {
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    const double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

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

inline uint64_t pack_bits(const std::vector<uint8_t>& bits) {
    uint64_t out = 0;
    for (int i = 0; i < (int)bits.size(); ++i)
        if (bits[i] & 1u) out |= (uint64_t{1} << i);
    return out;
}

struct DistributionChange {
    double kl_current_to_next = 0.0;
    double kl_next_to_current = 0.0;
    double js = 0.0;
};

void contract_axis(std::vector<double>& dist,
                   int64_t flip,
                   double log0,
                   double log1);

DistributionChange divergence_between_dists(
    const std::vector<double>& current,
    const std::vector<double>& next)
{
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    const double logz_current = logsumexp_all(current);
    const double logz_next = logsumexp_all(next);
    if (logz_current == NEG_INF || logz_next == NEG_INF)
        return {};

    DistributionChange out;
    bool inf_current_to_next = false;
    bool inf_next_to_current = false;
    const double log_half = std::log(0.5);

    for (int64_t state = 0; state < (int64_t)current.size(); ++state) {
        const double lp = current[(size_t)state] - logz_current;
        const double lq = next[(size_t)state] - logz_next;

        const bool p_fin = std::isfinite(lp);
        const bool q_fin = std::isfinite(lq);
        if (p_fin) {
            const double p = std::exp(lp);
            if (q_fin) out.kl_current_to_next += p * (lp - lq);
            else       inf_current_to_next = true;
            const double lm = logsumexp(lp, lq) + log_half;
            out.js += 0.5 * p * (lp - lm);
        }
        if (q_fin) {
            const double q = std::exp(lq);
            if (p_fin) out.kl_next_to_current += q * (lq - lp);
            else       inf_next_to_current = true;
            const double lm = logsumexp(lp, lq) + log_half;
            out.js += 0.5 * q * (lq - lm);
        }
    }

    if (inf_current_to_next)
        out.kl_current_to_next = std::numeric_limits<double>::infinity();
    if (inf_next_to_current)
        out.kl_next_to_current = std::numeric_limits<double>::infinity();
    return out;
}

DistributionChange contraction_change_from_dist(
    const std::vector<double>& dist,
    int64_t                    flip,
    double                     log0,
    double                     log1)
{
    std::vector<double> next = dist;
    contract_axis(next, flip, log0, log1);
    auto out = divergence_between_dists(dist, next);
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

std::vector<double> initial_class_slice(int num_observables,
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

void contract_axis(std::vector<double>& dist,
                   int64_t flip,
                   double log0,
                   double log1)
{
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    std::vector<double> next(dist.size(), NEG_INF);
    for (int64_t state = 0; state < (int64_t)dist.size(); ++state) {
        next[(size_t)state] = logsumexp(
            next[(size_t)state], dist[(size_t)state] + log0);
        const int64_t state1 = state ^ flip;
        next[(size_t)state1] = logsumexp(
            next[(size_t)state1], dist[(size_t)state] + log1);
    }
    dist = std::move(next);
}

std::pair<std::vector<int64_t>, std::vector<double>>
top_states(const std::vector<double>& dist) {
    const int64_t keep = std::min<int64_t>(RECORD_MAX_STATES, (int64_t)dist.size());
    std::vector<int64_t> order((size_t)dist.size());
    for (int64_t state = 0; state < (int64_t)dist.size(); ++state)
        order[(size_t)state] = state;
    std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                      [&](int64_t a, int64_t b) {
                          return dist[(size_t)a] > dist[(size_t)b];
                      });
    std::vector<int64_t> states((size_t)keep);
    std::vector<double> logp((size_t)keep);
    for (int64_t i = 0; i < keep; ++i) {
        states[(size_t)i] = order[(size_t)i];
        logp[(size_t)i] = dist[(size_t)order[(size_t)i]];
    }
    return {std::move(states), std::move(logp)};
}

} // namespace

SurpriseMLDecoder::SurpriseMLDecoder(const FlashBPBase& bp,
                                     SurpriseMLLogger  logger,
                                     int               bond_dim)
    : num_detectors_(bp.num_detectors)
    , num_errors_(bp.num_errors)
    , num_observables_(bp.num_observables)
    , bond_dim_(bond_dim)
    , H_(bp.H_raw())
    , L_(bp.L_raw())
    , logger_(std::move(logger))
{
    const auto& probs = bp.error_probs_raw();
    log_p_.resize(num_errors_);
    log_1mp_.resize(num_errors_);
    for (int e = 0; e < num_errors_; ++e) {
        const double p = std::max(1e-300, std::min(1.0 - 1e-300, probs[e]));
        log_p_[e] = std::log(p);
        log_1mp_[e] = std::log(1.0 - p);
    }
    logger_("SurpriseMLDecoder constructed  errors=" +
            std::to_string(num_errors_) + "  observables=" +
            std::to_string(num_observables_) + "  bond_dim=" +
            std::to_string(bond_dim_), 1);
}

std::vector<double> SurpriseMLDecoder::class_log_probs(
    const std::vector<uint8_t>& syndrome) const
{
    std::vector<double> split;
    if (class_log_probs_split(syndrome, split))
        return split;
    return class_log_probs_dense(syndrome);
}

std::vector<double> SurpriseMLDecoder::class_log_probs_dense(
    const std::vector<uint8_t>& syndrome) const
{
    const int state_bits = num_detectors_ + num_observables_;
    constexpr int MAX_STATE_BITS = 30;
    if (state_bits > MAX_STATE_BITS)
        throw std::runtime_error(
            "SurpriseMLDecoder: dense path would allocate 2^" +
            std::to_string(state_bits) + " states; use a split code.");
    const int64_t n_states = int64_t{1} << state_bits;
    const int n_classes = 1 << num_observables_;
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
    const uint64_t syndrome_state = pack_bits(syndrome);
    logger_.record_ml_start(syndrome, num_detectors_, num_observables_,
                            "surprise-cpu");
    logger_.record_ml_step(
        -1, -1, 0, state_bits, n_states,
        std::vector<int64_t>{0}, std::vector<double>{0.0},
        initial_class_slice(num_observables_, syndrome_state));

    std::vector<uint8_t> remaining((size_t)num_errors_, 1);
    for (int order_pos = 0; order_pos < num_errors_; ++order_pos) {
        std::vector<double> scores((size_t)num_errors_, -1.0);
        DistributionChange best_div;
        int best_e = -1;
        for (int e = 0; e < num_errors_; ++e) {
            if (!remaining[(size_t)e]) continue;
            const DistributionChange div = contraction_change_from_dist(
                dist, flips[e], log_1mp_[e], log_p_[e]);
            scores[(size_t)e] = div.js;
            if (best_e < 0 || div.js > best_div.js ||
                (div.js == best_div.js && e < best_e)) {
                best_e = e;
                best_div = div;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        contract_axis(dist, flips[best_e], log_1mp_[best_e], log_p_[best_e]);
        remaining[(size_t)best_e] = 0;
        const auto t1 = std::chrono::steady_clock::now();
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        auto [states, logp] = top_states(dist);
        logger_("SurpriseML contraction axis=" + std::to_string(best_e) +
                "  order=" + std::to_string(order_pos) +
                "  change_js=" + std::to_string(best_div.js), 2);
        logger_.record_ml_step(
            order_pos, best_e, us, state_bits, n_states,
            std::move(states), std::move(logp),
            class_slice_from_dist(dist, num_detectors_, num_observables_,
                                  syndrome_state),
            std::move(scores),
            best_div.kl_current_to_next, best_div.kl_next_to_current,
            best_div.js);
    }

    std::vector<double> out((size_t)n_classes, NEG_INF);
    for (int cls = 0; cls < n_classes; ++cls) {
        const uint64_t state =
            syndrome_state | ((uint64_t)cls << num_detectors_);
        out[(size_t)cls] = dist[(size_t)state];
    }
    return out;
}

bool SurpriseMLDecoder::class_log_probs_split(
    const std::vector<uint8_t>& syndrome,
    std::vector<double>& out_logp) const
{
    const int total_nodes = num_errors_ + num_detectors_;
    std::vector<std::vector<int>> adj(total_nodes);
    for (int d = 0; d < num_detectors_; ++d) {
        for (int e = 0; e < num_errors_; ++e) {
            if (!H_[(size_t)d * num_errors_ + e]) continue;
            adj[e].push_back(num_errors_ + d);
            adj[num_errors_ + d].push_back(e);
        }
    }
    struct Component { std::vector<int> errors; std::vector<int> detectors; };
    std::vector<Component> comps;
    std::vector<uint8_t> seen(total_nodes, 0);
    for (int start = 0; start < total_nodes; ++start) {
        if (seen[start] || adj[start].empty()) continue;
        Component c;
        std::queue<int> q;
        seen[start] = 1;
        q.push(start);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (u < num_errors_) c.errors.push_back(u);
            else c.detectors.push_back(u - num_errors_);
            for (int v : adj[u]) {
                if (seen[v]) continue;
                seen[v] = 1;
                q.push(v);
            }
        }
        comps.push_back(std::move(c));
    }
    if (comps.size() <= 1)
        return false;

    logger_("SurpriseML decoding split components=" + std::to_string(comps.size()), 1);
    const int n_classes = 1 << num_observables_;
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    struct RuntimeComponent {
        std::vector<int> errors;
        std::vector<int> detectors;
        int state_bits = 0;
        int64_t states = 0;
        uint64_t target = 0;
        std::vector<int64_t> flips;
        std::vector<double> dist;
        std::vector<uint8_t> remaining;
        bool touched = false;
    };

    std::vector<RuntimeComponent> rt;
    rt.reserve(comps.size());
    int remaining_total = 0;
    for (const auto& comp : comps) {
        RuntimeComponent c;
        c.errors = comp.errors;
        c.detectors = comp.detectors;
        const int cd = (int)c.detectors.size();
        c.state_bits = cd + num_observables_;
        if (c.state_bits >= 63)
            throw std::runtime_error("SurpriseMLDecoder: split component too large.");
        c.states = int64_t{1} << c.state_bits;
        c.dist.assign((size_t)c.states, NEG_INF);
        c.dist[0] = 0.0;
        c.flips.assign(c.errors.size(), 0);
        c.remaining.assign(c.errors.size(), 1);
        remaining_total += (int)c.errors.size();

        for (size_t i = 0; i < c.errors.size(); ++i) {
            const int e = c.errors[i];
            uint64_t flip = 0;
            for (int ld = 0; ld < cd; ++ld)
                if (H_[(size_t)c.detectors[ld] * num_errors_ + e])
                    flip |= (uint64_t{1} << ld);
            for (int o = 0; o < num_observables_; ++o)
                if (L_[(size_t)o * num_errors_ + e])
                    flip |= (uint64_t{1} << (cd + o));
            c.flips[i] = (int64_t)flip;
        }
        for (int ld = 0; ld < cd; ++ld)
            if (syndrome[c.detectors[ld]] & 1u)
                c.target |= (uint64_t{1} << ld);
        rt.push_back(std::move(c));
    }

    auto visible_class_slice = [&]() {
        std::vector<double> acc((size_t)n_classes, NEG_INF);
        acc[0] = 0.0;
        for (const auto& c : rt) {
            if (!c.touched) continue;
            const int cd = (int)c.detectors.size();
            acc = xor_convolve_classes(
                acc, class_slice_from_dist(c.dist, cd,
                                           num_observables_, c.target));
        }
        return acc;
    };

    auto local_states_to_global = [&](const RuntimeComponent& c,
                                      const std::vector<int64_t>& local_states) {
        const int cd = (int)c.detectors.size();
        std::vector<int64_t> states(local_states.size(), 0);
        for (size_t i = 0; i < local_states.size(); ++i) {
            const int64_t local_state = local_states[i];
            uint64_t global = 0;
            for (int ld = 0; ld < cd; ++ld)
                if ((local_state >> ld) & 1)
                    global |= (uint64_t{1} << c.detectors[ld]);
            for (int o = 0; o < num_observables_; ++o)
                if ((local_state >> (cd + o)) & 1)
                    global |= (uint64_t{1} << (num_detectors_ + o));
            states[i] = (int64_t)global;
        }
        return states;
    };

    logger_.record_ml_start(syndrome, num_detectors_, num_observables_,
                            "surprise-split-cpu");
    logger_.record_ml_step(
        -1, -1, 0, rt.front().state_bits, rt.front().states,
        std::vector<int64_t>{0}, std::vector<double>{0.0},
        visible_class_slice());

    for (int order_pos = 0; order_pos < remaining_total; ++order_pos) {
        std::vector<double> scores((size_t)num_errors_, -1.0);
        DistributionChange best_div;
        int best_ci = -1;
        int best_local = -1;
        int best_e = -1;

        for (size_t ci = 0; ci < rt.size(); ++ci) {
            auto& c = rt[ci];
            for (size_t i = 0; i < c.errors.size(); ++i) {
                if (!c.remaining[i]) continue;
                const int e = c.errors[i];
                DistributionChange div = contraction_change_from_dist(
                    c.dist, c.flips[i], log_1mp_[e], log_p_[e]);
                scores[(size_t)e] = div.js;
                if (best_e < 0 || div.js > best_div.js ||
                    (div.js == best_div.js && e < best_e)) {
                    best_ci = (int)ci;
                    best_local = (int)i;
                    best_e = e;
                    best_div = div;
                }
            }
        }

        auto& chosen = rt[(size_t)best_ci];
        const auto t0 = std::chrono::steady_clock::now();
        contract_axis(chosen.dist, chosen.flips[(size_t)best_local],
                      log_1mp_[best_e], log_p_[best_e]);
        chosen.remaining[(size_t)best_local] = 0;
        chosen.touched = true;
        const auto t1 = std::chrono::steady_clock::now();
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        auto [local_states, logp] = top_states(chosen.dist);
        auto states = local_states_to_global(chosen, local_states);

        logger_("SurpriseML split global axis=" + std::to_string(best_e) +
                "  component=" + std::to_string(best_ci) +
                "  change_js=" + std::to_string(best_div.js), 2);
        logger_.record_ml_step(
            order_pos, best_e, us, chosen.state_bits, chosen.states,
            std::move(states), std::move(logp),
            visible_class_slice(),
            std::move(scores),
            best_div.kl_current_to_next, best_div.kl_next_to_current,
            best_div.js);
    }

    out_logp.assign((size_t)n_classes, NEG_INF);
    out_logp[0] = 0.0;
    for (const auto& c : rt) {
        const int cd = (int)c.detectors.size();
        out_logp = xor_convolve_classes(
            out_logp, class_slice_from_dist(c.dist, cd,
                                            num_observables_, c.target));
    }
    return true;
}

std::vector<uint8_t> SurpriseMLDecoder::find_representative(
    const std::vector<uint8_t>& syndrome,
    int target_class) const
{
    const int rows = num_detectors_ + num_observables_;
    const int cols = num_errors_;
    std::vector<std::vector<uint8_t>> A(
        rows, std::vector<uint8_t>((size_t)cols + 1, 0));
    for (int d = 0; d < num_detectors_; ++d) {
        for (int e = 0; e < cols; ++e)
            A[d][e] = H_[(size_t)d * cols + e] & 1u;
        A[d][cols] = syndrome[d] & 1u;
    }
    for (int o = 0; o < num_observables_; ++o) {
        int r = num_detectors_ + o;
        for (int e = 0; e < cols; ++e)
            A[r][e] = L_[(size_t)o * cols + e] & 1u;
        A[r][cols] = (uint8_t)((target_class >> o) & 1);
    }
    int pivot_row = 0;
    std::vector<int> pivot_cols;
    for (int col = 0; col < cols && pivot_row < rows; ++col) {
        int found = -1;
        for (int r = pivot_row; r < rows; ++r)
            if (A[r][col]) { found = r; break; }
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
        bool zero = true;
        for (int c = 0; c < cols; ++c)
            if (A[r][c]) { zero = false; break; }
        if (zero && A[r][cols])
            throw std::runtime_error("SurpriseMLDecoder: no representative.");
    }
    std::vector<uint8_t> x(cols, 0);
    for (int r = 0; r < (int)pivot_cols.size(); ++r)
        x[pivot_cols[r]] = A[r][cols] & 1u;
    return x;
}

std::vector<uint8_t> SurpriseMLDecoder::operator()(
    const std::vector<uint8_t>& syndrome,
    int /*max_iter*/)
{
    if ((int)syndrome.size() != num_detectors_)
        throw std::invalid_argument("Syndrome length must equal num_detectors.");
    logger_.set_shot(shot_counter_);
    logger_.set_iteration(0);
    ++shot_counter_;

    const auto logp = class_log_probs(syndrome);
    const int best = (int)std::distance(
        logp.begin(), std::max_element(logp.begin(), logp.end()));
    logger_.record_ml_final(logp, best);
    logger_("SurpriseML class chosen=" + std::to_string(best) +
            "  log_p=" + std::to_string(logp[(size_t)best]), 2);
    return find_representative(syndrome, best);
}
