#pragma once
#include "logger.h"
#include <iomanip>
#include <sstream>

// ---------------------------------------------------------------------------
// DecodeLogger<false> — zero-overhead no-op, same as Logger<false>.
// Setter calls are empty inlines and compile away entirely.
// ---------------------------------------------------------------------------
template<bool Enabled>
struct DecodeLogger : public Logger<Enabled> {
    inline void set_batch(unsigned int)     noexcept {}
    inline void set_shot(unsigned int)      noexcept {}
    inline void set_iteration(unsigned int) noexcept {}
};

// ---------------------------------------------------------------------------
// DecodeLogger<true> — Logger<true> extended with decode context.
//
// batch / shot / iteration are stored in a shared_ptr<DecodeState> so every
// copy of the logger (e.g. FlashBP's and the decoder's) reads the same values.
// Calling set_batch() on either copy is immediately visible to the other.
//
// Log lines are prepended with:  [BBBB:SSSS:IIII]  (zero-padded to 4 digits)
// ---------------------------------------------------------------------------
template<>
struct DecodeLogger<true> : public Logger<true> {
private:
    struct DecodeState {
        unsigned int batch     = 0;
        unsigned int shot      = 0;
        unsigned int iteration = 0;
    };

    std::shared_ptr<DecodeState> decode_state_;

    std::string make_prefix() const {
        std::ostringstream oss;
        oss << '['
            << std::setw(4) << std::setfill('0') << decode_state_->batch     << ':'
            << std::setw(4) << std::setfill('0') << decode_state_->shot      << ':'
            << std::setw(4) << std::setfill('0') << decode_state_->iteration
            << "] ";
        return oss.str();
    }

public:
    DecodeLogger(unsigned int       lvl,
                 bool               console,
                 bool               buffered,
                 const std::string& log_file)
        : Logger<true>(lvl, console, buffered, log_file)
        , decode_state_(std::make_shared<DecodeState>())
    {}

    void set_batch(unsigned int b)     { decode_state_->batch     = b; }
    void set_shot(unsigned int s)      { decode_state_->shot      = s; }
    void set_iteration(unsigned int i) { decode_state_->iteration = i; }

    void operator()(const std::string& msg, unsigned int msg_level) const {
        if (this->level < msg_level) return;
        const std::string formatted = make_prefix() + msg;
        if (this->buffered)
            this->state_->buffer.emplace_back(msg_level, formatted);
        else
            this->write(formatted, msg_level);
    }
};
