#pragma once

#include "record_logger.h"
#include "../decoders/gbp_region_policy.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct GBPRegionRecord {
    int center_check = -1;
    int activation = 0;
    std::vector<int> data;
    std::vector<int> cycle_checks;
    std::vector<int> axis_edge;
    std::vector<int> output_edges;
    std::vector<int> output_axes;
    std::vector<int> internal_check_indices;
    std::vector<uint32_t> internal_check_masks;
    uint64_t dense_state_count = 0;
    int64_t valid_state_count = -1;
};

struct GBPRecordingMetadata {
    std::string policy;
    std::string backend;
    int degree = 0;
    int num_detectors = 0;
    int num_errors = 0;
    int num_edges = 0;
    std::vector<GBPRegionRecord> regions;
};

struct GBPLogger : public RecordLogger {
private:
    struct GBPState {
        GBPRecordingMetadata metadata;
        bool has_metadata = false;
    };

    std::shared_ptr<GBPState> gbp_state_;

public:
    GBPLogger(unsigned int lvl,
              bool console,
              bool buffered,
              const std::string& log_file)
        : RecordLogger(lvl, console, buffered, log_file)
        , gbp_state_(std::make_shared<GBPState>())
    {}

    void record_gbp_start(const std::string& policy,
                          const std::string& backend,
                          int degree,
                          int num_detectors,
                          int num_errors,
                          int num_edges,
                          const std::vector<GBPRegion>& regions)
    {
        GBPRecordingMetadata meta;
        meta.policy = policy;
        meta.backend = backend;
        meta.degree = degree;
        meta.num_detectors = num_detectors;
        meta.num_errors = num_errors;
        meta.num_edges = num_edges;
        meta.regions.reserve(regions.size());

        for (const auto& region : regions) {
            GBPRegionRecord rec;
            rec.center_check = region.center_check;
            rec.activation = static_cast<int>(region.activation);
            rec.data = region.data;
            rec.cycle_checks = region.cycle_checks;
            rec.axis_edge = region.axis_edge;
            rec.output_edges.reserve(region.outputs.size());
            rec.output_axes.reserve(region.outputs.size());
            for (const auto& out : region.outputs) {
                rec.output_edges.push_back(out.edge_idx);
                rec.output_axes.push_back(out.axis);
            }
            rec.internal_check_indices.reserve(region.internal_checks.size());
            rec.internal_check_masks.reserve(region.internal_checks.size());
            for (const auto& ic : region.internal_checks) {
                rec.internal_check_indices.push_back(ic.check_idx);
                rec.internal_check_masks.push_back(ic.mask);
            }
            rec.dense_state_count =
                region.data.size() >= 63
                ? 0
                : (uint64_t{1} << region.data.size());
            meta.regions.push_back(std::move(rec));
        }

        gbp_state_->metadata = std::move(meta);
        gbp_state_->has_metadata = true;
    }

    void record_gbp_iteration(int iter,
                              const std::vector<uint8_t>& syndrome,
                              const std::vector<uint8_t>& decision,
                              const std::vector<double>& msg_v2c,
                              const std::vector<double>& msg_c2v,
                              const std::vector<int>& active_regions)
    {
        record_iteration(iter, syndrome, decision, msg_v2c, msg_c2v);
        if (record_state_->shots.empty()) return;
        auto& iterations = record_state_->shots.back().iterations;
        if (iterations.empty()) return;
        iterations.back().active_regions = active_regions;
    }

    const GBPRecordingMetadata& gbp_metadata() const {
        return gbp_state_->metadata;
    }

    bool has_gbp_metadata() const {
        return gbp_state_->has_metadata;
    }
};
