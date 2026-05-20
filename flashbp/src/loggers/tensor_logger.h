#pragma once
#include "record_logger.h"
#include <utility>

// ---------------------------------------------------------------------------
// TensorLogger — RecordLogger extended with per-check tensor capture.
//
// TensorDecoder calls record_tensor(c, ...) once per check c during the
// check-node update phase; the tensors are stashed in shared state.
// When record_iteration is called at the end of the iteration, the stash
// is moved into IterationRecord.tensors and cleared.
//
// All state lives in shared_ptrs so FlashBP's logger copy and the decoder's
// logger copy share the same store.
// ---------------------------------------------------------------------------
struct TensorLogger : public RecordLogger {
private:
    struct TensorState {
        std::vector<CheckTensor> pending;
    };

    std::shared_ptr<TensorState> tensor_state_;

public:
    TensorLogger(unsigned int       lvl,
                 bool               console,
                 bool               buffered,
                 const std::string& log_file)
        : RecordLogger(lvl, console, buffered, log_file)
        , tensor_state_(std::make_shared<TensorState>())
    {}

    // Stash one check's tensor for the current iteration.
    // `weight` and `parity` are moved in; caller need not keep them alive.
    void record_tensor(int                  check_idx,
                       std::vector<int>     nbhd_data,
                       std::vector<double>  weight,
                       std::vector<uint8_t> parity)
    {
        // Optional verbose summary at level 4
        (*this)("tensor c=" + std::to_string(check_idx) +
                " axes=" + std::to_string(nbhd_data.size()) +
                " configs=" + std::to_string(weight.size()), 4);

        tensor_state_->pending.push_back(
            {check_idx, std::move(nbhd_data),
             std::move(weight), std::move(parity)});
    }

    // Shadows RecordLogger::record_iteration. Pushes the iteration record via
    // the parent, then moves the pending tensors into the just-created entry.
    void record_iteration(int                          iter,
                          const std::vector<uint8_t>& syndrome,
                          const std::vector<uint8_t>& decision,
                          const std::vector<double>&  msg_v2c,
                          const std::vector<double>&  msg_c2v)
    {
        RecordLogger::record_iteration(iter, syndrome, decision,
                                       msg_v2c, msg_c2v);

        auto& shots = record_state_->shots;
        if (!shots.empty() && !shots.back().iterations.empty()) {
            shots.back().iterations.back().tensors =
                std::move(tensor_state_->pending);
        }
        tensor_state_->pending.clear();
    }
};
