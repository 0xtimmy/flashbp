#pragma once
#include "decoder.h"
#include "../loggers/ml_logger.h"
#include <cstdint>
#include <vector>

class FlashBPBase;

class SurpriseMLDecoder : public Decoder {
public:
    explicit SurpriseMLDecoder(const FlashBPBase& bp,
                               SurpriseMLLogger  logger,
                               int               bond_dim);

    std::vector<uint8_t> operator()(
        const std::vector<uint8_t>& syndrome,
        int max_iter = 1
    ) override;

    int num_detectors() const override { return num_detectors_; }
    int num_errors()    const override { return num_errors_; }

private:
    std::vector<double> class_log_probs(const std::vector<uint8_t>& syndrome) const;
    bool class_log_probs_split(const std::vector<uint8_t>& syndrome,
                               std::vector<double>& out_logp) const;
    std::vector<double> class_log_probs_dense(const std::vector<uint8_t>& syndrome) const;
    std::vector<uint8_t> find_representative(const std::vector<uint8_t>& syndrome,
                                             int target_class) const;

    int num_detectors_;
    int num_errors_;
    int num_observables_;
    int bond_dim_;

    std::vector<uint8_t> H_;
    std::vector<uint8_t> L_;
    std::vector<double>  log_p_;
    std::vector<double>  log_1mp_;

    SurpriseMLLogger logger_;
    unsigned int shot_counter_ = 0;
};
