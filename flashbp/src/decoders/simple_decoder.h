#pragma once
#include "decoder.h"
#include "../loggers/tensor_logger.h"
#include "../loggers/ml_logger.h"
#include <vector>
#include <cstdint>

class FlashBPBase;

template<typename LoggerT>
class SimpleDecoder : public Decoder {
public:
    explicit SimpleDecoder(const FlashBPBase& bp, LoggerT logger);

    std::vector<uint8_t> operator()(
        const std::vector<uint8_t>& syndrome,
        int max_iter = 100
    ) override;

    int num_detectors() const override { return num_detectors_; }
    int num_errors()    const override { return num_errors_; }

private:
    int num_detectors_;
    int num_errors_;

    struct Edge { int check, var; };
    std::vector<Edge>             edges_;
    std::vector<std::vector<int>> var_edges_;
    std::vector<std::vector<int>> check_edges_;
    std::vector<double>           ch_llr_;

    LoggerT logger_;
    unsigned int shot_counter_ = 0;
};

// Suppress implicit instantiation in other TUs — definitions live in simple_decoder.cpp
extern template class SimpleDecoder<Logger<false>>;
extern template class SimpleDecoder<Logger<true>>;
extern template class SimpleDecoder<DecodeLogger<true>>;
extern template class SimpleDecoder<RecordLogger>;
extern template class SimpleDecoder<TensorLogger>;
extern template class SimpleDecoder<MLLogger>;
