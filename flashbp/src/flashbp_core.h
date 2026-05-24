#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <cstdint>

#include "loggers/logger.h"
#include "loggers/decode_logger.h"
#include "loggers/record_logger.h"
#include "loggers/tensor_logger.h"
#include "loggers/ml_logger.h"
#include "loggers/gbp_logger.h"

class Decoder;

// ── Abstract base — this is what Python sees ──────────────────────────────────

class FlashBPBase {
public:
    int num_detectors   = 0;
    int num_errors      = 0;
    int num_observables = 0;

    virtual ~FlashBPBase() = default;

    virtual pybind11::array_t<uint8_t> decode(pybind11::array_t<uint8_t> syndrome,
                                              int max_iter) const = 0;
    virtual void flush() = 0;
    virtual pybind11::dict last_decode_stats() const = 0;

    // Decode-context setters — no-op by default, active for DecodeLogger<true> and subclasses
    virtual void set_batch(unsigned int)     noexcept {}
    virtual void set_shot(unsigned int)      noexcept {}
    virtual void set_iteration(unsigned int) noexcept {}

    virtual pybind11::array_t<uint8_t> get_H()           const = 0;
    virtual pybind11::array_t<uint8_t> get_L()           const = 0;
    virtual pybind11::array_t<double>  get_error_probs() const = 0;

    // Returns recorded iteration data; py::none() unless log_type="record"
    virtual pybind11::object get_recording() const { return pybind11::none(); }

    const std::vector<uint8_t>& H_raw()           const { return H_; }
    const std::vector<uint8_t>& L_raw()           const { return L_; }
    const std::vector<double>&  error_probs_raw() const { return error_probs_; }

protected:
    std::vector<double>  error_probs_;
    std::vector<uint8_t> H_;
    std::vector<uint8_t> L_;
};

// ── Concrete template — logger baked in at compile time ───────────────────────

template<typename LoggerT>
class FlashBP : public FlashBPBase {
public:
    explicit FlashBP(pybind11::object dem, pybind11::object config);
    ~FlashBP() override;

    pybind11::array_t<uint8_t> decode(pybind11::array_t<uint8_t> syndrome,
                                      int max_iter) const override;
    void flush() override { logger_.flush(); }
    pybind11::dict last_decode_stats() const override;

    // Forward context setters to any logger derived from DecodeLogger<true>
    void set_batch(unsigned int b) noexcept override {
        if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>)
            logger_.set_batch(b);
    }
    void set_shot(unsigned int s) noexcept override {
        if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>)
            logger_.set_shot(s);
    }
    void set_iteration(unsigned int i) noexcept override {
        if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>)
            logger_.set_iteration(i);
    }

    pybind11::array_t<uint8_t> get_H()           const override;
    pybind11::array_t<uint8_t> get_L()           const override;
    pybind11::array_t<double>  get_error_probs() const override;

    pybind11::object get_recording() const override;

    const LoggerT& logger() const { return logger_; }

private:
    LoggerT                  logger_;
    std::unique_ptr<Decoder> decoder_;
};

extern template class FlashBP<Logger<false>>;
extern template class FlashBP<Logger<true>>;
extern template class FlashBP<DecodeLogger<true>>;
extern template class FlashBP<RecordLogger>;
extern template class FlashBP<TensorLogger>;
extern template class FlashBP<GBPLogger>;
extern template class FlashBP<MLLogger>;
extern template class FlashBP<SurpriseMLLogger>;

std::unique_ptr<FlashBPBase> make_flashbp(pybind11::object dem,
                                          pybind11::object config);
