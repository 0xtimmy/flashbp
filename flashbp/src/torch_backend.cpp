#include "torch_backend.h"

#include <sstream>

#ifdef FLASHBP_HAS_TORCH
  #include <torch/torch.h>
#endif

namespace flashbp::torch_backend {

bool available() {
#ifdef FLASHBP_HAS_TORCH
    return true;
#else
    return false;
#endif
}

#ifdef FLASHBP_HAS_TORCH
namespace {

// Run a tiny GEMM on a device and return a one-line status string.
std::string smoke_test_on(torch::Device dev) {
    std::ostringstream os;
    try {
        const auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(dev);
        auto A = torch::ones({4, 4}, opts);
        auto C = torch::mm(A, A);
        double s = C.sum().item<double>();
        os << "OK  (4x4 ones·ones).sum() = " << s
           << " (expected 64)";
    } catch (const std::exception& e) {
        os << "FAILED — " << e.what();
    }
    return os.str();
}

} // anonymous
#endif

std::string diagnostics() {
    std::ostringstream os;

#ifdef FLASHBP_HAS_TORCH
    os << "LibTorch enabled\n";
    os << "  version      : " << TORCH_VERSION << "\n";
    os << "  CUDA built   : " << (torch::cuda::is_available() ? "yes" : "no") << "\n";

    // CPU smoke test always runs
    os << "  CPU smoke    : "
       << smoke_test_on(torch::Device(torch::kCPU)) << "\n";

    // Per-device smoke tests when a GPU runtime is present.  Under either
    // a CUDA build or a ROCm build of LibTorch the device kind is kCUDA;
    // ROCm masquerades as CUDA in the API.
    if (torch::cuda::is_available()) {
        const auto n = (int)torch::cuda::device_count();
        os << "  GPU devices  : " << n << "\n";
        for (int i = 0; i < n; ++i) {
            os << "  GPU[" << i << "] smoke : "
               << smoke_test_on(torch::Device(torch::kCUDA, i)) << "\n";
        }
    } else {
        os << "  GPU devices  : 0  (no CUDA / ROCm runtime detected)\n";
    }
#else
    os << "LibTorch NOT enabled.\n"
       << "  Rebuild with LibTorch on CMAKE_PREFIX_PATH, e.g.\n"
       << "    pip install . --no-build-isolation \\\n"
       << "      -C cmake.args=\"-DCMAKE_PREFIX_PATH=/path/to/libtorch\"\n";
#endif

    return os.str();
}

} // namespace flashbp::torch_backend
