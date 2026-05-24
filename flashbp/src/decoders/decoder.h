#pragma once
#include <vector>
#include <cstdint>

class Decoder {
public:
    virtual ~Decoder() = default;

    virtual std::vector<uint8_t> operator()(
        const std::vector<uint8_t>& syndrome,
        int max_iter = 100
    ) = 0;

    virtual int num_detectors() const = 0;
    virtual int num_errors()    const = 0;

    bool last_converged() const { return last_converged_; }
    int last_iterations() const { return last_iterations_; }

protected:
    void reset_decode_stats() {
        last_converged_ = false;
        last_iterations_ = 0;
    }

    void set_decode_stats(bool converged, int iterations) {
        last_converged_ = converged;
        last_iterations_ = iterations;
    }

private:
    bool last_converged_ = false;
    int last_iterations_ = 0;
};
