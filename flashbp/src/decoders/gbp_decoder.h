#pragma once

#include "decoder.h"
#include "gbp_region_policy.h"
#include "../loggers/tensor_logger.h"
#include "../loggers/ml_logger.h"
#include <cstdint>
#include <memory>
#include <vector>

class FlashBPBase;

template<typename LoggerT>
class GBPDecoder : public Decoder {
public:
    static constexpr int MAX_AXES = 22;

    explicit GBPDecoder(const FlashBPBase& bp,
                        LoggerT logger,
                        int degree,
                        std::unique_ptr<RegionGroupingPolicy> policy);

    std::vector<uint8_t> operator()(
        const std::vector<uint8_t>& syndrome,
        int max_iter = 100
    ) override;

    int num_detectors() const override { return num_detectors_; }
    int num_errors() const override { return num_errors_; }

private:
    int num_detectors_;
    int num_errors_;
    int degree_;

    std::vector<TannerEdge> edges_;
    std::vector<std::vector<int>> var_edges_;
    std::vector<std::vector<int>> check_edges_;
    std::vector<double> ch_llr_;

    std::unique_ptr<RegionGroupingPolicy> policy_;
    std::vector<GBPRegion> regions_;
    std::vector<int> edge_axis_pos_;

    LoggerT logger_;
    unsigned int shot_counter_ = 0;
};

extern template class GBPDecoder<Logger<false>>;
extern template class GBPDecoder<Logger<true>>;
extern template class GBPDecoder<DecodeLogger<true>>;
extern template class GBPDecoder<RecordLogger>;
extern template class GBPDecoder<TensorLogger>;
extern template class GBPDecoder<MLLogger>;
