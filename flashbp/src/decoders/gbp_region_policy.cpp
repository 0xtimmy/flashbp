#include "gbp_region_policy.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace {

void populate_region_checks_and_outputs(
    GBPRegion& region,
    const std::set<int>& output_checks,
    int num_errors,
    const std::vector<TannerEdge>& edges,
    const std::vector<std::vector<int>>& var_edges,
    const std::vector<std::vector<int>>& check_edges
) {
    std::vector<int> axis_of_v(num_errors, -1);
    for (int k = 0; k < (int)region.data.size(); ++k)
        axis_of_v[region.data[k]] = k;

    region.axis_edge.assign(region.data.size(), -1);
    region.outputs.clear();
    for (int c : output_checks) {
        for (int ei : check_edges[c]) {
            int k = axis_of_v[edges[ei].var];
            if (k < 0) continue;
            region.outputs.push_back({ei, k});
            if (region.axis_edge[k] < 0)
                region.axis_edge[k] = ei;
        }
    }

    std::unordered_set<int> candidate_checks;
    for (int v : region.data) {
        for (int ei : var_edges[v])
            candidate_checks.insert(edges[ei].check);
    }

    region.internal_checks.clear();
    for (int cc : candidate_checks) {
        uint32_t mask = 0;
        bool fully_in = true;
        for (int ei : check_edges[cc]) {
            int k = axis_of_v[edges[ei].var];
            if (k < 0) {
                fully_in = false;
                break;
            }
            mask |= (uint32_t)1 << k;
        }
        if (fully_in)
            region.internal_checks.push_back({mask, cc});
    }
}

std::vector<std::vector<int>> canonical_simple_cycles(
    int num_vars,
    int num_checks,
    const std::vector<TannerEdge>& edges,
    int max_length
) {
    const int num_nodes = num_vars + num_checks;
    std::vector<std::vector<int>> adj(num_nodes);
    for (const auto& edge : edges) {
        const int v = edge.var;
        const int c = num_vars + edge.check;
        adj[v].push_back(c);
        adj[c].push_back(v);
    }
    for (auto& ns : adj) {
        std::sort(ns.begin(), ns.end());
        ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
    }

    std::set<std::vector<int>> unique_cycles;
    std::vector<int> path;
    std::vector<uint8_t> in_path(num_nodes, 0);

    auto canonicalize = [](std::vector<int> cycle) {
        std::vector<int> best;
        for (int pass = 0; pass < 2; ++pass) {
            if (pass == 1)
                std::reverse(cycle.begin(), cycle.end());
            for (int i = 0; i < (int)cycle.size(); ++i) {
                std::vector<int> rotated;
                rotated.reserve(cycle.size());
                for (int j = 0; j < (int)cycle.size(); ++j)
                    rotated.push_back(cycle[(i + j) % cycle.size()]);
                if (best.empty() || rotated < best)
                    best = std::move(rotated);
            }
        }
        return best;
    };

    std::function<void(int, int)> dfs = [&](int start, int node) {
        if ((int)path.size() > max_length)
            return;
        for (int nbr : adj[node]) {
            if (nbr == start && path.size() >= 4) {
                unique_cycles.insert(canonicalize(path));
                continue;
            }
            if (nbr <= start || in_path[nbr])
                continue;
            if ((int)path.size() >= max_length)
                continue;
            in_path[nbr] = 1;
            path.push_back(nbr);
            dfs(start, nbr);
            path.pop_back();
            in_path[nbr] = 0;
        }
    };

    for (int start = 0; start < num_nodes; ++start) {
        path.clear();
        std::fill(in_path.begin(), in_path.end(), 0);
        path.push_back(start);
        in_path[start] = 1;
        dfs(start, start);
    }

    std::vector<std::vector<int>> cycles(unique_cycles.begin(), unique_cycles.end());
    std::sort(cycles.begin(), cycles.end(),
              [](const auto& a, const auto& b) {
                  if (a.size() != b.size()) return a.size() < b.size();
                  return a < b;
              });
    return cycles;
}

GBPRegion make_single_check_region(
    int check,
    int num_errors,
    const std::vector<TannerEdge>& edges,
    const std::vector<std::vector<int>>& var_edges,
    const std::vector<std::vector<int>>& check_edges
) {
    GBPRegion region;
    region.center_check = check;
    std::set<int> data;
    for (int ei : check_edges[check])
        data.insert(edges[ei].var);
    region.data.assign(data.begin(), data.end());
    populate_region_checks_and_outputs(
        region, std::set<int>{check}, num_errors, edges, var_edges, check_edges);
    return region;
}

struct CycleSeed {
    std::set<int> checks;
    std::set<int> data;
};

class DisjointSet {
public:
    explicit DisjointSet(int n) : parent_(n), rank_(n, 0) {
        for (int i = 0; i < n; ++i) parent_[i] = i;
    }

    int find(int x) {
        if (parent_[x] != x) parent_[x] = find(parent_[x]);
        return parent_[x];
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) return;
        if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
        parent_[rb] = ra;
        if (rank_[ra] == rank_[rb]) rank_[ra] += 1;
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

void warn_region_too_wide(const std::string& context,
                          int axes,
                          const GBPRegionBudget& budget,
                          uint64_t estimated_states) {
    std::cerr << "WARNING: " << context << " produced region with "
              << axes << " axes, estimated_states=" << estimated_states
              << " > max_states=" << budget.max_states
              << " or axes > max_axes=" << budget.max_axes << "; "
              << "using a smaller region or skipping it." << std::endl;
}

int binary_rank(std::vector<uint32_t> rows) {
    int rank = 0;
    for (int bit = 31; bit >= 0; --bit) {
        int pivot = -1;
        for (int r = rank; r < (int)rows.size(); ++r) {
            if ((rows[r] >> bit) & 1u) {
                pivot = r;
                break;
            }
        }
        if (pivot < 0) continue;
        std::swap(rows[rank], rows[pivot]);
        for (int r = 0; r < (int)rows.size(); ++r) {
            if (r != rank && ((rows[r] >> bit) & 1u))
                rows[r] ^= rows[rank];
        }
        ++rank;
    }
    return rank;
}

uint64_t estimate_region_states(const GBPRegion& region,
                                const GBPRegionBudget& budget) {
    const int K = (int)region.data.size();
    if (K >= 63) return UINT64_MAX;
    if (!budget.use_valid_state_estimate)
        return uint64_t{1} << K;

    std::vector<uint32_t> masks;
    masks.reserve(region.internal_checks.size());
    for (const auto& ic : region.internal_checks)
        if (ic.mask != 0) masks.push_back(ic.mask);
    const int rank = binary_rank(std::move(masks));
    const int free_axes = std::max(0, K - rank);
    return free_axes >= 63 ? UINT64_MAX : (uint64_t{1} << free_axes);
}

bool region_exceeds_budget(const GBPRegion& region,
                           const GBPRegionBudget& budget,
                           uint64_t* estimated_states = nullptr) {
    const uint64_t states = estimate_region_states(region, budget);
    if (estimated_states) *estimated_states = states;
    return (int)region.data.size() > budget.max_axes ||
           states > budget.max_states;
}

} // anonymous namespace

CheckNeighborhoodPolicy::CheckNeighborhoodPolicy(int degree, GBPRegionBudget budget)
    : degree_(degree)
    , budget_(budget)
{
    if (degree < 1)
        throw std::invalid_argument("CheckNeighborhoodPolicy: degree must be >= 1.");
}

std::vector<GBPRegion> CheckNeighborhoodPolicy::build_regions(
    int num_detectors,
    int num_errors,
    const std::vector<TannerEdge>& edges,
    const std::vector<std::vector<int>>& var_edges,
    const std::vector<std::vector<int>>& check_edges,
    std::vector<int>& edge_axis_pos
) const {
    std::vector<GBPRegion> regions(num_detectors);
    edge_axis_pos.assign(edges.size(), -1);

    for (int c = 0; c < num_detectors; ++c) {
        std::set<int> data_in_region;
        std::set<int> visited_checks{c};
        std::vector<int> frontier_checks{c};

        for (int hop = 0; hop < degree_; ++hop) {
            std::vector<int> new_data;
            for (int cc : frontier_checks) {
                for (int ei : check_edges[cc]) {
                    int v = edges[ei].var;
                    if (data_in_region.insert(v).second)
                        new_data.push_back(v);
                }
            }
            if (hop + 1 == degree_) break;

            std::vector<int> new_checks;
            for (int v : new_data) {
                for (int ei : var_edges[v]) {
                    int cc = edges[ei].check;
                    if (visited_checks.insert(cc).second)
                        new_checks.push_back(cc);
                }
            }
            if (new_checks.empty()) break;
            frontier_checks = std::move(new_checks);
        }

        GBPRegion region;
        region.center_check = c;
        region.data.assign(data_in_region.begin(), data_in_region.end());

        uint64_t estimated_states = 0;
        populate_region_checks_and_outputs(
            region, std::set<int>{c}, num_errors, edges, var_edges, check_edges);
        if (region_exceeds_budget(region, budget_, &estimated_states)) {
            warn_region_too_wide(
                "CheckNeighborhoodPolicy check " + std::to_string(c) +
                " at degree=" + std::to_string(degree_),
                (int)region.data.size(),
                budget_,
                estimated_states);
            region = make_single_check_region(
                c, num_errors, edges, var_edges, check_edges);
            if (region_exceeds_budget(region, budget_, &estimated_states)) {
                warn_region_too_wide(
                    "CheckNeighborhoodPolicy fallback check " + std::to_string(c),
                    (int)region.data.size(),
                    budget_,
                    estimated_states);
                regions[c] = GBPRegion{};
                continue;
            }
        }
        for (const auto& out : region.outputs)
            edge_axis_pos[out.edge_idx] = out.axis;

        regions[c] = std::move(region);
    }

    return regions;
}

ShortCyclePolicy::ShortCyclePolicy(int max_length,
                                   GBPRegionBudget budget,
                                   GBPRegionActivation activation,
                                   bool union_overlaps,
                                   std::string name)
    : max_length_(max_length < 4 ? 8 : max_length)
    , budget_(budget)
    , activation_(activation)
    , union_overlaps_(union_overlaps)
    , name_(std::move(name))
{}

std::vector<GBPRegion> ShortCyclePolicy::build_regions(
    int num_detectors,
    int num_errors,
    const std::vector<TannerEdge>& edges,
    const std::vector<std::vector<int>>& var_edges,
    const std::vector<std::vector<int>>& check_edges,
    std::vector<int>& edge_axis_pos
) const {
    edge_axis_pos.assign(edges.size(), -1);
    std::vector<GBPRegion> regions;
    std::vector<uint8_t> check_covered(num_detectors, 0);
    std::vector<CycleSeed> seeds;

    auto cycles = canonical_simple_cycles(num_errors, num_detectors, edges, max_length_);
    for (const auto& cycle : cycles) {
        std::set<int> cycle_checks;
        for (int node : cycle) {
            if (node >= num_errors)
                cycle_checks.insert(node - num_errors);
        }
        if (cycle_checks.empty())
            continue;

        std::set<int> data;
        for (int c : cycle_checks) {
            if (activation_ == GBPRegionActivation::Always)
                check_covered[c] = 1;
            for (int ei : check_edges[c])
                data.insert(edges[ei].var);
        }
        seeds.push_back({std::move(cycle_checks), std::move(data)});
    }

    if (union_overlaps_ && !seeds.empty()) {
        std::vector<CycleSeed> original_seeds = seeds;
        DisjointSet dsu((int)seeds.size());
        std::vector<int> owner_check(num_detectors, -1);
        std::vector<int> owner_data(num_errors, -1);
        for (int i = 0; i < (int)seeds.size(); ++i) {
            for (int c : seeds[i].checks) {
                if (owner_check[c] >= 0) dsu.unite(i, owner_check[c]);
                else owner_check[c] = i;
            }
            for (int v : seeds[i].data) {
                if (owner_data[v] >= 0) dsu.unite(i, owner_data[v]);
                else owner_data[v] = i;
            }
        }

        std::vector<CycleSeed> merged(seeds.size());
        std::vector<uint8_t> used(seeds.size(), 0);
        for (int i = 0; i < (int)seeds.size(); ++i) {
            const int root = dsu.find(i);
            used[root] = 1;
            merged[root].checks.insert(seeds[i].checks.begin(), seeds[i].checks.end());
            merged[root].data.insert(seeds[i].data.begin(), seeds[i].data.end());
        }

        std::vector<CycleSeed> compact;
        for (int i = 0; i < (int)merged.size(); ++i)
            if (used[i]) compact.push_back(std::move(merged[i]));
        bool oversized_union = false;
        int max_union_axes = 0;
        uint64_t max_union_states = 0;
        for (const auto& seed : compact) {
            GBPRegion candidate;
            candidate.center_check = *seed.checks.begin();
            candidate.cycle_checks.assign(seed.checks.begin(), seed.checks.end());
            candidate.data.assign(seed.data.begin(), seed.data.end());
            populate_region_checks_and_outputs(
                candidate, seed.checks, num_errors, edges, var_edges, check_edges);
            uint64_t estimated_states = 0;
            max_union_axes = std::max(max_union_axes, (int)candidate.data.size());
            if (region_exceeds_budget(candidate, budget_, &estimated_states))
                oversized_union = true;
            max_union_states = std::max(max_union_states, estimated_states);
        }
        if (oversized_union) {
            warn_region_too_wide(
                "ShortCyclePolicy union component; disabling union for this policy",
                max_union_axes,
                budget_,
                max_union_states);
            seeds = std::move(original_seeds);
        } else {
            seeds = std::move(compact);
        }
    }

    for (const auto& seed : seeds) {
        GBPRegion region;
        region.center_check = *seed.checks.begin();
        region.activation = activation_;
        region.cycle_checks.assign(seed.checks.begin(), seed.checks.end());
        region.data.assign(seed.data.begin(), seed.data.end());

        populate_region_checks_and_outputs(
            region, seed.checks, num_errors, edges, var_edges, check_edges);
        uint64_t estimated_states = 0;
        if (region_exceeds_budget(region, budget_, &estimated_states)) {
            warn_region_too_wide(
                "ShortCyclePolicy cycle at max_length=" +
                std::to_string(max_length_),
                (int)region.data.size(),
                budget_,
                estimated_states);
            continue;
        }

        if (!region.outputs.empty())
            regions.push_back(std::move(region));
    }

    const bool add_all_single_check_regions =
        activation_ != GBPRegionActivation::Always;
    for (int c = 0; c < num_detectors; ++c) {
        if (!add_all_single_check_regions && check_covered[c])
            continue;
        GBPRegion region = make_single_check_region(
            c, num_errors, edges, var_edges, check_edges);
        uint64_t estimated_states = 0;
        if (region_exceeds_budget(region, budget_, &estimated_states)) {
            warn_region_too_wide(
                "ShortCyclePolicy fallback check region " + std::to_string(c),
                (int)region.data.size(),
                budget_,
                estimated_states);
            continue;
        }
        regions.push_back(std::move(region));
    }

    for (int ri = 0; ri < (int)regions.size(); ++ri) {
        for (const auto& out : regions[ri].outputs) {
            if (edge_axis_pos[out.edge_idx] < 0)
                edge_axis_pos[out.edge_idx] = out.axis;
        }
    }

    return regions;
}

ManualGroupPolicy::ManualGroupPolicy(std::vector<GBPManualGroup> groups,
                                     GBPRegionBudget budget,
                                     bool add_single_check_regions)
    : groups_(std::move(groups))
    , budget_(budget)
    , add_single_check_regions_(add_single_check_regions)
{}

std::vector<GBPRegion> ManualGroupPolicy::build_regions(
    int num_detectors,
    int num_errors,
    const std::vector<TannerEdge>& edges,
    const std::vector<std::vector<int>>& var_edges,
    const std::vector<std::vector<int>>& check_edges,
    std::vector<int>& edge_axis_pos
) const {
    edge_axis_pos.assign(edges.size(), -1);
    std::vector<GBPRegion> regions;

    for (int gi = 0; gi < (int)groups_.size(); ++gi) {
        const auto& group = groups_[gi];
        std::set<int> data;
        std::set<int> checks;
        for (int v : group.data) {
            if (v < 0 || v >= num_errors)
                throw std::invalid_argument(
                    "ManualGroupPolicy data index out of range: " +
                    std::to_string(v));
            data.insert(v);
        }
        for (int c : group.checks) {
            if (c < 0 || c >= num_detectors)
                throw std::invalid_argument(
                    "ManualGroupPolicy check index out of range: " +
                    std::to_string(c));
            checks.insert(c);
        }
        if (data.empty() || checks.empty())
            throw std::invalid_argument(
                "ManualGroupPolicy groups require non-empty data and checks.");

        GBPRegion region;
        region.center_check = group.center_check >= 0
                              ? group.center_check
                              : *checks.begin();
        region.activation = group.activation;
        region.data.assign(data.begin(), data.end());
        region.cycle_checks.assign(checks.begin(), checks.end());
        populate_region_checks_and_outputs(
            region, checks, num_errors, edges, var_edges, check_edges);
        uint64_t estimated_states = 0;
        if (region_exceeds_budget(region, budget_, &estimated_states)) {
            warn_region_too_wide(
                "ManualGroupPolicy group " + std::to_string(gi),
                (int)region.data.size(),
                budget_,
                estimated_states);
            continue;
        }
        if (!region.outputs.empty())
            regions.push_back(std::move(region));
    }

    if (add_single_check_regions_) {
        for (int c = 0; c < num_detectors; ++c) {
            GBPRegion region = make_single_check_region(
                c, num_errors, edges, var_edges, check_edges);
            uint64_t estimated_states = 0;
            if (region_exceeds_budget(region, budget_, &estimated_states)) {
                warn_region_too_wide(
                    "ManualGroupPolicy fallback check region " +
                    std::to_string(c),
                    (int)region.data.size(),
                    budget_,
                    estimated_states);
                continue;
            }
            regions.push_back(std::move(region));
        }
    }

    for (int ri = 0; ri < (int)regions.size(); ++ri) {
        for (const auto& out : regions[ri].outputs) {
            if (edge_axis_pos[out.edge_idx] < 0)
                edge_axis_pos[out.edge_idx] = out.axis;
        }
    }
    return regions;
}

std::unique_ptr<RegionGroupingPolicy> make_region_grouping_policy(
    const std::string& policy,
    int degree,
    GBPRegionBudget budget,
    std::vector<GBPManualGroup> manual_groups,
    bool manual_add_single_check_regions
) {
    if (policy == "manual" || policy == "manual_groups" ||
        policy == "fixed_groups") {
        return std::make_unique<ManualGroupPolicy>(
            std::move(manual_groups), budget, manual_add_single_check_regions);
    }
    if (policy.empty() || policy == "check" || policy == "check_neighborhood")
        return std::make_unique<CheckNeighborhoodPolicy>(degree, budget);
    if (policy == "cycles" || policy == "short_cycles")
        return std::make_unique<ShortCyclePolicy>(
            degree, budget, GBPRegionActivation::Always, false, "short_cycles");
    if (policy == "cycles_any_active" ||
        policy == "short_cycles_any_active" ||
        policy == "cycles_any_on" ||
        policy == "short_cycles_any_on")
        return std::make_unique<ShortCyclePolicy>(
            degree, budget, GBPRegionActivation::AnyCheckActive, false,
            "short_cycles_any_active");
    if (policy == "cycles_all_active" ||
        policy == "short_cycles_all_active" ||
        policy == "cycles_all_on" ||
        policy == "short_cycles_all_on")
        return std::make_unique<ShortCyclePolicy>(
            degree, budget, GBPRegionActivation::AllChecksActive, false,
            "short_cycles_all_active");
    if (policy == "union_cycles" ||
        policy == "short_cycles_union" ||
        policy == "union_short_cycles")
        return std::make_unique<ShortCyclePolicy>(
            degree, budget, GBPRegionActivation::Always, true,
            "short_cycles_union");
    if (policy == "union_cycles_any_active" ||
        policy == "short_cycles_union_any_active" ||
        policy == "union_short_cycles_any_active" ||
        policy == "union_cycles_any_on" ||
        policy == "short_cycles_union_any_on")
        return std::make_unique<ShortCyclePolicy>(
            degree, budget, GBPRegionActivation::AnyCheckActive, true,
            "short_cycles_union_any_active");
    if (policy == "union_cycles_all_active" ||
        policy == "short_cycles_union_all_active" ||
        policy == "union_short_cycles_all_active" ||
        policy == "union_cycles_all_on" ||
        policy == "short_cycles_union_all_on")
        return std::make_unique<ShortCyclePolicy>(
            degree, budget, GBPRegionActivation::AllChecksActive, true,
            "short_cycles_union_all_active");

    throw std::invalid_argument(
        "Unknown GBP region_policy: \"" + policy +
        "\". Available: \"check_neighborhood\", \"manual_groups\", "
        "\"short_cycles\", "
        "\"short_cycles_any_active\", \"short_cycles_all_active\", "
        "\"short_cycles_union\", \"short_cycles_union_any_active\", "
        "\"short_cycles_union_all_active\".");
}
