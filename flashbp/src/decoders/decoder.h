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
};
