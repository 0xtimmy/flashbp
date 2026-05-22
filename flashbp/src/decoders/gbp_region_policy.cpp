#include "gbp_region_policy.h"

#include <algorithm>
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

} // anonymous namespace

CheckNeighborhoodPolicy::CheckNeighborhoodPolicy(int degree, int max_axes)
    : degree_(degree)
    , max_axes_(max_axes)
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

        const int K = (int)region.data.size();
        if (K > max_axes_) {
            throw std::runtime_error(
                "CheckNeighborhoodPolicy: check " + std::to_string(c) +
                " produced region with " + std::to_string(K) +
                " axes > max_axes=" + std::to_string(max_axes_) +
                " at degree=" + std::to_string(degree_) + ".");
        }

        populate_region_checks_and_outputs(
            region, std::set<int>{c}, num_errors, edges, var_edges, check_edges);
        for (const auto& out : region.outputs)
            edge_axis_pos[out.edge_idx] = out.axis;

        regions[c] = std::move(region);
    }

    return regions;
}

ShortCyclePolicy::ShortCyclePolicy(int max_length,
                                   int max_axes,
                                   GBPRegionActivation activation,
                                   bool union_overlaps,
                                   std::string name)
    : max_length_(max_length < 4 ? 8 : max_length)
    , max_axes_(max_axes)
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
        seeds = std::move(compact);
    }

    for (const auto& seed : seeds) {
        GBPRegion region;
        region.center_check = *seed.checks.begin();
        region.activation = activation_;
        region.cycle_checks.assign(seed.checks.begin(), seed.checks.end());
        region.data.assign(seed.data.begin(), seed.data.end());
        if ((int)region.data.size() > max_axes_) {
            throw std::runtime_error(
                "ShortCyclePolicy: cycle region has " +
                std::to_string(region.data.size()) +
                " axes > max_axes=" + std::to_string(max_axes_) +
                " at max_length=" + std::to_string(max_length_) + ".");
        }

        populate_region_checks_and_outputs(
            region, seed.checks, num_errors, edges, var_edges, check_edges);
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
        if ((int)region.data.size() > max_axes_) {
            throw std::runtime_error(
                "ShortCyclePolicy: fallback check region " + std::to_string(c) +
                " has " + std::to_string(region.data.size()) +
                " axes > max_axes=" + std::to_string(max_axes_) + ".");
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

std::unique_ptr<RegionGroupingPolicy> make_region_grouping_policy(
    const std::string& policy,
    int degree,
    int max_axes
) {
    if (policy.empty() || policy == "check" || policy == "check_neighborhood")
        return std::make_unique<CheckNeighborhoodPolicy>(degree, max_axes);
    if (policy == "cycles" || policy == "short_cycles")
        return std::make_unique<ShortCyclePolicy>(
            degree, max_axes, GBPRegionActivation::Always, false, "short_cycles");
    if (policy == "cycles_any_active" ||
        policy == "short_cycles_any_active" ||
        policy == "cycles_any_on" ||
        policy == "short_cycles_any_on")
        return std::make_unique<ShortCyclePolicy>(
            degree, max_axes, GBPRegionActivation::AnyCheckActive, false,
            "short_cycles_any_active");
    if (policy == "cycles_all_active" ||
        policy == "short_cycles_all_active" ||
        policy == "cycles_all_on" ||
        policy == "short_cycles_all_on")
        return std::make_unique<ShortCyclePolicy>(
            degree, max_axes, GBPRegionActivation::AllChecksActive, false,
            "short_cycles_all_active");
    if (policy == "union_cycles" ||
        policy == "short_cycles_union" ||
        policy == "union_short_cycles")
        return std::make_unique<ShortCyclePolicy>(
            degree, max_axes, GBPRegionActivation::Always, true,
            "short_cycles_union");
    if (policy == "union_cycles_any_active" ||
        policy == "short_cycles_union_any_active" ||
        policy == "union_short_cycles_any_active" ||
        policy == "union_cycles_any_on" ||
        policy == "short_cycles_union_any_on")
        return std::make_unique<ShortCyclePolicy>(
            degree, max_axes, GBPRegionActivation::AnyCheckActive, true,
            "short_cycles_union_any_active");
    if (policy == "union_cycles_all_active" ||
        policy == "short_cycles_union_all_active" ||
        policy == "union_short_cycles_all_active" ||
        policy == "union_cycles_all_on" ||
        policy == "short_cycles_union_all_on")
        return std::make_unique<ShortCyclePolicy>(
            degree, max_axes, GBPRegionActivation::AllChecksActive, true,
            "short_cycles_union_all_active");

    throw std::invalid_argument(
        "Unknown GBP region_policy: \"" + policy +
        "\". Available: \"check_neighborhood\", \"short_cycles\", "
        "\"short_cycles_any_active\", \"short_cycles_all_active\", "
        "\"short_cycles_union\", \"short_cycles_union_any_active\", "
        "\"short_cycles_union_all_active\".");
}
