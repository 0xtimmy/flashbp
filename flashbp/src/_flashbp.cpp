#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "flashbp_core.h"
#include "torch_backend.h"

namespace py = pybind11;

PYBIND11_MODULE(_flashbp, m) {
    m.doc() = "Accelerated belief propagation decoder for quantum error correction.";

    py::class_<FlashBPBase>(m, "FlashBP")
        .def(py::init([](py::object dem, py::object config) {
                return make_flashbp(dem, config);
             }),
             py::arg("dem"),
             py::arg("config"),
             "Construct from a stim.DetectorErrorModel and a DecoderConfig.")
        .def_readonly("num_detectors",   &FlashBPBase::num_detectors)
        .def_readonly("num_errors",      &FlashBPBase::num_errors)
        .def_readonly("num_observables", &FlashBPBase::num_observables)
        .def_property_readonly("H",           &FlashBPBase::get_H,
             "Parity-check matrix (num_detectors x num_errors), uint8.")
        .def_property_readonly("L",           &FlashBPBase::get_L,
             "Observable matrix (num_observables x num_errors), uint8.")
        .def_property_readonly("error_probs", &FlashBPBase::get_error_probs,
             "Prior error probabilities (num_errors,), float64.")
        .def("decode", &FlashBPBase::decode,
             py::arg("syndrome"),
             py::arg("max_iter") = 100,
             "Run the selected decoder on a syndrome (uint8, length num_detectors).\n"
             "Returns hard-decision error vector (uint8, length num_errors).")
        .def("flush", &FlashBPBase::flush,
             "Flush buffered log messages to their configured sinks.\n"
             "No-op when log=False or log_buffered=False.")
        .def("set_batch",     &FlashBPBase::set_batch,     py::arg("batch"),
             "Set the batch index prepended to DecodeLogger log lines. No-op otherwise.")
        .def("set_shot",      &FlashBPBase::set_shot,      py::arg("shot"),
             "Set the shot index prepended to DecodeLogger log lines. No-op otherwise.")
        .def("set_iteration", &FlashBPBase::set_iteration, py::arg("iteration"),
             "Set the iteration index prepended to DecodeLogger log lines. No-op otherwise.")
        .def("get_recording", &FlashBPBase::get_recording,
             "Return recorded per-iteration data as a list of shot dicts.\n"
             "Each dict has keys: batch, shot, iterations.\n"
             "Each iteration dict has: iteration, syndrome, decision, msg_v2c, msg_c2v.\n"
             "Returns None unless log_type='record'.");

    // ── LibTorch backend introspection ──────────────────────────────────────
    m.def("torch_available", &flashbp::torch_backend::available,
          "True if the binary was built with LibTorch linked in.");
    m.def("torch_diagnostics", &flashbp::torch_backend::diagnostics,
          "Multi-line diagnostic string: Torch version, CUDA/ROCm availability,\n"
          "device count, and per-device GEMM smoke test.");
}
