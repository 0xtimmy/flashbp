#pragma once

#include "gbp_region_policy.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class GBPBackend {
public:
    virtual ~GBPBackend() = default;

    virtual const char* name() const = 0;

    virtual void prepare(const std::vector<GBPRegion>& regions,
                         const std::vector<double>& ch_llr) = 0;

    virtual void update_regions(
        const std::vector<uint8_t>& syndrome,
        const std::vector<double>& msg_v2c,
        std::vector<double>& next_c2v,
        std::vector<int>& next_c2v_count) = 0;
};

class DenseGBPBackend : public GBPBackend {
public:
    const char* name() const override { return "dense_cpu"; }

    void prepare(const std::vector<GBPRegion>& regions,
                 const std::vector<double>& ch_llr) override;

    void update_regions(
        const std::vector<uint8_t>& syndrome,
        const std::vector<double>& msg_v2c,
        std::vector<double>& next_c2v,
        std::vector<int>& next_c2v_count) override;

protected:
    const std::vector<GBPRegion>* regions_ = nullptr;
    const std::vector<double>* ch_llr_ = nullptr;
    std::vector<double> weight_;
    std::vector<uint8_t> bad_;
};

class SparseGBPBackend : public DenseGBPBackend {
public:
    const char* name() const override { return "sparse_cpu"; }

    void prepare(const std::vector<GBPRegion>& regions,
                 const std::vector<double>& ch_llr) override;

    void update_regions(
        const std::vector<uint8_t>& syndrome,
        const std::vector<double>& msg_v2c,
        std::vector<double>& next_c2v,
        std::vector<int>& next_c2v_count) override;

private:
    struct RegionCache {
        std::unordered_map<uint32_t, std::vector<uint32_t>> states_by_syndrome;
    };

    std::vector<RegionCache> cache_;
    std::vector<double> sparse_weight_;
};

std::unique_ptr<GBPBackend> make_gbp_backend(const std::string& backend);
