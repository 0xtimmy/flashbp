#pragma once
#include "decode_logger.h"
#include <cstdint>
#include <memory>
#include <vector>

// Per-check tensor snapshot — populated only by TensorLogger.
struct CheckTensor {
    int                  check_idx;
    std::vector<int>     nbhd_data;   // sorted data-node indices on the tensor axes
    std::vector<double>  weight;      // 2^|nbhd_data| flat tensor
    std::vector<uint8_t> parity;      // same shape; 0 = valid, 1 = violates a local check
};

struct IterationRecord {
    int                       iteration;
    std::vector<uint8_t>      syndrome;
    std::vector<uint8_t>      decision;
    std::vector<double>       msg_v2c;
    std::vector<double>       msg_c2v;
    std::vector<CheckTensor>  tensors;   // empty unless using TensorLogger
};

struct ShotRecord {
    unsigned int                 batch;
    unsigned int                 shot;
    std::vector<IterationRecord> iterations;
};

// ---------------------------------------------------------------------------
// RecordLogger — DecodeLogger<true> extended with per-iteration data capture.
//
// Calling set_shot() automatically opens a new ShotRecord.  The decoder calls
// record_iteration() at the end of each BP iteration.  All state lives in a
// shared_ptr<RecordState> so FlashBP's copy and the decoder's copy both write
// to the same store.
// ---------------------------------------------------------------------------
struct RecordLogger : public DecodeLogger<true> {
protected:
    struct RecordState {
        unsigned int          current_batch = 0;
        std::vector<ShotRecord> shots;
    };

    std::shared_ptr<RecordState> record_state_;

public:
    RecordLogger(unsigned int       lvl,
                 bool               console,
                 bool               buffered,
                 const std::string& log_file)
        : DecodeLogger<true>(lvl, console, buffered, log_file)
        , record_state_(std::make_shared<RecordState>())
    {}

    // Override setters to keep current_batch in shared state and auto-open shots
    void set_batch(unsigned int b) {
        DecodeLogger<true>::set_batch(b);
        record_state_->current_batch = b;
    }

    void set_shot(unsigned int s) {
        DecodeLogger<true>::set_shot(s);
        record_state_->shots.push_back({record_state_->current_batch, s, {}});
    }

    void record_iteration(int                          iter,
                          const std::vector<uint8_t>& syndrome,
                          const std::vector<uint8_t>& decision,
                          const std::vector<double>&  msg_v2c,
                          const std::vector<double>&  msg_c2v)
    {
        if (record_state_->shots.empty()) return;
        record_state_->shots.back().iterations.push_back(
            {iter, syndrome, decision, msg_v2c, msg_c2v, {}});
    }

    const std::vector<ShotRecord>& shots() const { return record_state_->shots; }
};
