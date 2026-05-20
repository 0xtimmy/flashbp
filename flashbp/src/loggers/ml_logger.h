#pragma once
#include "decode_logger.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct MLContractionStep {
    int                  axis;
    int                  error_idx;
    long long            duration_us;
    int                  state_bits;
    long long            num_states;
    std::vector<int64_t> states;
    std::vector<double>  log_probs;
    std::vector<double>  class_log_probs;
};

struct MLShotRecord {
    unsigned int                   batch = 0;
    unsigned int                   shot = 0;
    std::vector<uint8_t>           syndrome;
    int                            num_detectors = 0;
    int                            num_observables = 0;
    std::string                    device;
    std::vector<MLContractionStep> steps;
    std::vector<double>            class_log_probs;
    int                            best_class = -1;
};

// ---------------------------------------------------------------------------
// MLLogger -- DecodeLogger<true> extended with maximum-likelihood contraction
// recording. State is shared so FlashBP and the decoder see the same store.
// ---------------------------------------------------------------------------
struct MLLogger : public DecodeLogger<true> {
private:
    struct MLState {
        unsigned int             current_batch = 0;
        std::vector<MLShotRecord> shots;
    };

    std::shared_ptr<MLState> ml_state_;

public:
    MLLogger(unsigned int       lvl,
             bool               console,
             bool               buffered,
             const std::string& log_file)
        : DecodeLogger<true>(lvl, console, buffered, log_file)
        , ml_state_(std::make_shared<MLState>())
    {}

    void set_batch(unsigned int b) {
        DecodeLogger<true>::set_batch(b);
        ml_state_->current_batch = b;
    }

    void set_shot(unsigned int s) {
        DecodeLogger<true>::set_shot(s);
        ml_state_->shots.push_back({ml_state_->current_batch, s});
    }

    void record_ml_start(const std::vector<uint8_t>& syndrome,
                         int                         num_detectors,
                         int                         num_observables,
                         const std::string&          device) const
    {
        if (ml_state_->shots.empty()) return;
        auto& shot = ml_state_->shots.back();
        shot.syndrome        = syndrome;
        shot.num_detectors   = num_detectors;
        shot.num_observables = num_observables;
        shot.device          = device;
        shot.steps.clear();
        shot.class_log_probs.clear();
        shot.best_class = -1;
    }

    void record_ml_step(int                         axis,
                        int                         error_idx,
                        long long                   duration_us,
                        int                         state_bits,
                        long long                   num_states,
                        std::vector<int64_t>        states,
                        std::vector<double>         log_probs,
                        std::vector<double>         class_log_probs = {}) const
    {
        if (ml_state_->shots.empty()) return;
        ml_state_->shots.back().steps.push_back(
            {axis, error_idx, duration_us, state_bits, num_states,
             std::move(states), std::move(log_probs),
             std::move(class_log_probs)});
    }

    void record_ml_final(std::vector<double> class_log_probs,
                         int                 best_class) const
    {
        if (ml_state_->shots.empty()) return;
        auto& shot = ml_state_->shots.back();
        shot.class_log_probs = std::move(class_log_probs);
        shot.best_class      = best_class;
    }

    const std::vector<MLShotRecord>& ml_shots() const {
        return ml_state_->shots;
    }
};
