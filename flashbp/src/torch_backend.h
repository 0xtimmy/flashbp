#pragma once
#include <string>

// ---------------------------------------------------------------------------
// LibTorch backend integration.
//
// Kept deliberately as a thin pimpl-style facade so that <torch/torch.h>
// (which is a *very* heavy include) only enters torch_backend.cpp.  The
// rest of flashbp can call these functions without depending on Torch.
//
// All functions degrade gracefully when built with -DFLASHBP_ENABLE_TORCH=OFF.
// ---------------------------------------------------------------------------

namespace flashbp::torch_backend {

// True iff the binary was compiled with LibTorch linked in.
bool available();

// Human-readable diagnostic string: torch version, device availability,
// device count, and a tiny GEMM smoke test on each available device.
// Useful for verifying CUDA / ROCm pickup at install time.
std::string diagnostics();

} // namespace flashbp::torch_backend
