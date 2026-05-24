#include "gbp_backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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

std::vector<uint32_t> generate_valid_states(
    int K,
    const std::vector<GBPInternalCheck>& checks,
    const std::vector<uint8_t>& syndrome)
{
    struct Row {
        uint32_t mask = 0;
        uint8_t rhs = 0;
    };

    std::vector<Row> rows;
    rows.reserve(checks.size());
    for (const auto& ic : checks) {
        if (ic.mask != 0)
            rows.push_back({ic.mask, (uint8_t)(syndrome[ic.check_idx] & 1)});
    }

    int rank = 0;
    std::vector<int> pivot_cols;
    pivot_cols.reserve(rows.size());
    for (int bit = 0; bit < K; ++bit) {
        int pivot = -1;
        for (int r = rank; r < (int)rows.size(); ++r) {
            if ((rows[r].mask >> bit) & 1u) {
                pivot = r;
                break;
            }
        }
        if (pivot < 0) continue;
        std::swap(rows[rank], rows[pivot]);
        for (int r = 0; r < (int)rows.size(); ++r) {
            if (r != rank && ((rows[r].mask >> bit) & 1u)) {
                rows[r].mask ^= rows[rank].mask;
                rows[r].rhs ^= rows[rank].rhs;
            }
        }
        pivot_cols.push_back(bit);
        ++rank;
    }

    for (const auto& row : rows) {
        if (row.mask == 0 && row.rhs)
            return {};
    }

    std::vector<uint8_t> is_pivot(K, 0);
    for (int bit : pivot_cols)
        is_pivot[bit] = 1;

    std::vector<int> free_cols;
    free_cols.reserve(K - rank);
    for (int bit = 0; bit < K; ++bit) {
        if (!is_pivot[bit])
            free_cols.push_back(bit);
    }

    const int F = (int)free_cols.size();
    const uint32_t total = uint32_t{1} << F;
    std::vector<uint32_t> states;
    states.reserve(total);

    for (uint32_t assign = 0; assign < total; ++assign) {
        uint32_t state = 0;
        for (int i = 0; i < F; ++i) {
            if ((assign >> i) & 1u)
                state |= uint32_t{1} << free_cols[i];
        }
        for (int r = rank - 1; r >= 0; --r) {
            const int pivot_bit = pivot_cols[r];
            const uint32_t without_pivot =
                rows[r].mask & ~(uint32_t{1} << pivot_bit);
            const int parity = popcount32(state & without_pivot) & 1;
            if ((parity ^ rows[r].rhs) & 1)
                state |= uint32_t{1} << pivot_bit;
        }
        states.push_back(state);
    }

    return states;
}

} // anonymous namespace

void DenseGBPBackend::prepare(const std::vector<GBPRegion>& regions,
                              const std::vector<double>& ch_llr) {
    regions_ = &regions;
    ch_llr_ = &ch_llr;

    size_t max_states = 1;
    for (const auto& region : regions)
        max_states = std::max(max_states, (size_t)1 << region.data.size());
    weight_.assign(max_states, 0.0);
    bad_.assign(max_states, 0);
}

void DenseGBPBackend::update_regions(
    const std::vector<uint8_t>& syndrome,
    const std::vector<double>& msg_v2c,
    std::vector<double>& next_c2v,
    std::vector<int>& next_c2v_count)
{
    if (!regions_ || !ch_llr_)
        throw std::logic_error("DenseGBPBackend used before prepare().");

    constexpr double INF = std::numeric_limits<double>::infinity();
    constexpr double SAT = 1e30;

    for (const GBPRegion& region : *regions_) {
        if (!region_is_active(region, syndrome))
            continue;
        const int K = (int)region.data.size();
        if (K == 0) continue;
        const size_t N = (size_t)1 << K;

        std::vector<double> incoming(K);
        for (int k = 0; k < K; ++k) {
            int ei = region.axis_edge[k];
            incoming[k] = (ei >= 0) ? msg_v2c[ei] : (*ch_llr_)[region.data[k]];
        }

        for (size_t idx = 0; idx < N; ++idx) {
            double w = 0.0;
            for (int k = 0; k < K; ++k) {
                int x = (int)((idx >> k) & 1);
                w += minsum_cost(x, incoming[k]);
            }
            weight_[idx] = w;
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
            bad_[idx] = violated;
        }

        for (const auto& out : region.outputs) {
            int ei = out.edge_idx;
            int k = out.axis;
            double l_v = msg_v2c[ei];

            double W0 = INF;
            double W1 = INF;
            for (size_t idx = 0; idx < N; ++idx) {
                if (bad_[idx]) continue;
                int x = (int)((idx >> k) & 1);
                double w_ext = weight_[idx] - minsum_cost(x, l_v);
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
}

void SparseGBPBackend::prepare(const std::vector<GBPRegion>& regions,
                               const std::vector<double>& ch_llr) {
    regions_ = &regions;
    ch_llr_ = &ch_llr;
    cache_.clear();
    cache_.resize(regions.size());

    for (size_t ri = 0; ri < regions.size(); ++ri) {
        const GBPRegion& region = regions[ri];
        const int K = (int)region.data.size();
        if (K >= 32)
            throw std::runtime_error(
                "SparseGBPBackend currently supports regions with at most 31 axes.");
        if (region.internal_checks.size() >= 32)
            throw std::runtime_error(
                "SparseGBPBackend currently supports regions with at most 31 internal checks.");
    }

    sparse_weight_.assign(1, 0.0);
}

void SparseGBPBackend::update_regions(
    const std::vector<uint8_t>& syndrome,
    const std::vector<double>& msg_v2c,
    std::vector<double>& next_c2v,
    std::vector<int>& next_c2v_count)
{
    if (!regions_ || !ch_llr_)
        throw std::logic_error("SparseGBPBackend used before prepare().");

    constexpr double INF = std::numeric_limits<double>::infinity();
    constexpr double SAT = 1e30;

    for (size_t ri = 0; ri < regions_->size(); ++ri) {
        const GBPRegion& region = (*regions_)[ri];
        if (!region_is_active(region, syndrome))
            continue;

        const int K = (int)region.data.size();
        if (K == 0) continue;

        uint32_t syndrome_key = 0;
        for (size_t ci = 0; ci < region.internal_checks.size(); ++ci) {
            const auto& ic = region.internal_checks[ci];
            syndrome_key |= (uint32_t)(syndrome[ic.check_idx] & 1) << ci;
        }

        auto& states_by_syndrome = cache_[ri].states_by_syndrome;
        auto found = states_by_syndrome.find(syndrome_key);
        if (found == states_by_syndrome.end()) {
            found = states_by_syndrome.emplace(
                syndrome_key,
                generate_valid_states(K, region.internal_checks, syndrome)).first;
        }
        if (found->second.empty())
            continue;
        const std::vector<uint32_t>& states = found->second;

        std::vector<double> incoming(K);
        for (int k = 0; k < K; ++k) {
            int ei = region.axis_edge[k];
            incoming[k] = (ei >= 0) ? msg_v2c[ei] : (*ch_llr_)[region.data[k]];
        }

        if (sparse_weight_.size() < states.size())
            sparse_weight_.resize(states.size());

        for (size_t si = 0; si < states.size(); ++si) {
            const uint32_t idx = states[si];
            double w = 0.0;
            for (int k = 0; k < K; ++k) {
                int x = (int)((idx >> k) & 1u);
                w += minsum_cost(x, incoming[k]);
            }
            sparse_weight_[si] = w;
        }

        for (const auto& out : region.outputs) {
            int ei = out.edge_idx;
            int k = out.axis;
            double l_v = msg_v2c[ei];

            double W0 = INF;
            double W1 = INF;
            for (size_t si = 0; si < states.size(); ++si) {
                const uint32_t idx = states[si];
                int x = (int)((idx >> k) & 1u);
                double w_ext = sparse_weight_[si] - minsum_cost(x, l_v);
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
}

std::unique_ptr<GBPBackend> make_gbp_backend(const std::string& backend) {
    if (backend.empty() ||
        backend == "cpu" ||
        backend == "dense" ||
        backend == "dense_cpu") {
        return std::make_unique<DenseGBPBackend>();
    }
    if (backend == "sparse" || backend == "sparse_cpu") {
        return std::make_unique<SparseGBPBackend>();
    }
    if (backend == "cuda" || backend == "torch" || backend == "torch_cuda") {
        throw std::runtime_error(
            "GBP backend '" + backend + "' is scaffolded but not implemented yet.");
    }
    throw std::invalid_argument(
        "Unknown GBP backend: \"" + backend +
        "\". Available: \"dense_cpu\", \"sparse_cpu\", \"torch_cuda\".");
}
