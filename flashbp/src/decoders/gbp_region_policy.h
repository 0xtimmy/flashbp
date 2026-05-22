#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct TannerEdge {
    int check;
    int var;
};

struct GBPInternalCheck {
    uint32_t mask;
    int check_idx;
};

struct GBPRegionOutput {
    int edge_idx;
    int axis;
};

enum class GBPRegionActivation {
    Always,
    AnyCheckActive,
    AllChecksActive,
};

struct GBPRegion {
    int center_check = -1;
    GBPRegionActivation activation = GBPRegionActivation::Always;
    std::vector<int> data;
    std::vector<int> cycle_checks;
    std::vector<int> axis_edge;
    std::vector<GBPRegionOutput> outputs;
    std::vector<GBPInternalCheck> internal_checks;
};

class RegionGroupingPolicy {
public:
    virtual ~RegionGroupingPolicy() = default;

    virtual std::vector<GBPRegion> build_regions(
        int num_detectors,
        int num_errors,
        const std::vector<TannerEdge>& edges,
        const std::vector<std::vector<int>>& var_edges,
        const std::vector<std::vector<int>>& check_edges,
        std::vector<int>& edge_axis_pos
    ) const = 0;

    virtual const char* name() const = 0;
};

class CheckNeighborhoodPolicy : public RegionGroupingPolicy {
public:
    CheckNeighborhoodPolicy(int degree, int max_axes);

    std::vector<GBPRegion> build_regions(
        int num_detectors,
        int num_errors,
        const std::vector<TannerEdge>& edges,
        const std::vector<std::vector<int>>& var_edges,
        const std::vector<std::vector<int>>& check_edges,
        std::vector<int>& edge_axis_pos
    ) const override;

    const char* name() const override { return "check_neighborhood"; }

private:
    int degree_;
    int max_axes_;
};

class ShortCyclePolicy : public RegionGroupingPolicy {
public:
    ShortCyclePolicy(int max_length,
                     int max_axes,
                     GBPRegionActivation activation,
                     bool union_overlaps,
                     std::string name);

    std::vector<GBPRegion> build_regions(
        int num_detectors,
        int num_errors,
        const std::vector<TannerEdge>& edges,
        const std::vector<std::vector<int>>& var_edges,
        const std::vector<std::vector<int>>& check_edges,
        std::vector<int>& edge_axis_pos
    ) const override;

    const char* name() const override { return name_.c_str(); }

private:
    int max_length_;
    int max_axes_;
    GBPRegionActivation activation_;
    bool union_overlaps_;
    std::string name_;
};

std::unique_ptr<RegionGroupingPolicy> make_region_grouping_policy(
    const std::string& policy,
    int degree,
    int max_axes
);
