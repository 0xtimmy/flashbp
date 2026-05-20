#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

// ---------------------------------------------------------------------------
// Logger<false> — zero-overhead no-op.
// Every call is an empty inline; the compiler removes it entirely.
// ---------------------------------------------------------------------------
template<bool Enabled>
struct Logger {
    unsigned int level = 0;

    inline void operator()(const std::string&, unsigned int) const noexcept {}
    inline void flush() noexcept {}
};

// ---------------------------------------------------------------------------
// Logger<true> — active logger.
//
// State (buffer + file handle) is shared_ptr so copies share the same sink.
// ---------------------------------------------------------------------------
template<>
struct Logger<true> {
protected:
    struct State {
        std::vector<std::pair<unsigned int, std::string>> buffer;
        std::unique_ptr<std::ofstream> file;
    };

    void write(const std::string& msg, unsigned int msg_level) const {
        const std::string line =
            "[" + std::to_string(msg_level) + "] " + msg + "\n";
        if (console)        std::cout << line;
        if (state_->file)  *state_->file << line;
    }

public:
    unsigned int           level;
    bool                   console;
    bool                   buffered;
    std::shared_ptr<State> state_;

    Logger(unsigned int        lvl,
           bool                console_,
           bool                buffered_,
           const std::string&  log_file)
        : level(lvl)
        , console(console_)
        , buffered(buffered_)
        , state_(std::make_shared<State>())
    {
        if (!log_file.empty())
            state_->file =
                std::make_unique<std::ofstream>(log_file, std::ios::app);
    }

    void operator()(const std::string& msg, unsigned int msg_level) const {
        if (level < msg_level) return;
        if (buffered)
            state_->buffer.emplace_back(msg_level, msg);
        else
            write(msg, msg_level);
    }

    void flush() {
        for (const auto& [lvl, msg] : state_->buffer)
            write(msg, lvl);
        state_->buffer.clear();
        if (state_->file) state_->file->flush();
    }
};
