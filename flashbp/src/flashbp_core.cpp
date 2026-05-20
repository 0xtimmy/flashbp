#include "flashbp_core.h"
#include "decoders/decoder.h"
#include "decoders/simple_decoder.h"
#include "decoders/tensor_decoder.h"
#include "decoders/fvs_decoder.h"
#include "decoders/degree_decoder.h"
#include "decoders/ml_decoder.h"
#include <stdexcept>
#include <string>

namespace py = pybind11;

// ── Logger construction from config ───────────────────────────────────────────

template<typename LoggerT>
static LoggerT make_logger(py::object /*config*/) { return LoggerT{}; }

template<>
Logger<true> make_logger<Logger<true>>(py::object config) {
    const unsigned int level    = py::cast<unsigned int>(config.attr("log_level"));
    const bool         console  = py::cast<bool>(config.attr("log_console"));
    const bool         buffered = py::cast<bool>(config.attr("log_buffered"));
    const std::string  file     = config.attr("log_file").is_none()
                                   ? std::string{}
                                   : py::cast<std::string>(config.attr("log_file"));
    return Logger<true>{level, console, buffered, file};
}

template<>
DecodeLogger<true> make_logger<DecodeLogger<true>>(py::object config) {
    const unsigned int level    = py::cast<unsigned int>(config.attr("log_level"));
    const bool         console  = py::cast<bool>(config.attr("log_console"));
    const bool         buffered = py::cast<bool>(config.attr("log_buffered"));
    const std::string  file     = config.attr("log_file").is_none()
                                   ? std::string{}
                                   : py::cast<std::string>(config.attr("log_file"));
    return DecodeLogger<true>{level, console, buffered, file};
}

template<>
RecordLogger make_logger<RecordLogger>(py::object config) {
    const unsigned int level    = py::cast<unsigned int>(config.attr("log_level"));
    const bool         console  = py::cast<bool>(config.attr("log_console"));
    const bool         buffered = py::cast<bool>(config.attr("log_buffered"));
    const std::string  file     = config.attr("log_file").is_none()
                                   ? std::string{}
                                   : py::cast<std::string>(config.attr("log_file"));
    return RecordLogger{level, console, buffered, file};
}

template<>
TensorLogger make_logger<TensorLogger>(py::object config) {
    const unsigned int level    = py::cast<unsigned int>(config.attr("log_level"));
    const bool         console  = py::cast<bool>(config.attr("log_console"));
    const bool         buffered = py::cast<bool>(config.attr("log_buffered"));
    const std::string  file     = config.attr("log_file").is_none()
                                   ? std::string{}
                                   : py::cast<std::string>(config.attr("log_file"));
    return TensorLogger{level, console, buffered, file};
}

template<>
MLLogger make_logger<MLLogger>(py::object config) {
    const unsigned int level    = py::cast<unsigned int>(config.attr("log_level"));
    const bool         console  = py::cast<bool>(config.attr("log_console"));
    const bool         buffered = py::cast<bool>(config.attr("log_buffered"));
    const std::string  file     = config.attr("log_file").is_none()
                                   ? std::string{}
                                   : py::cast<std::string>(config.attr("log_file"));
    return MLLogger{level, console, buffered, file};
}

// ── Internal decoder factory ──────────────────────────────────────────────────

template<typename LoggerT>
static std::unique_ptr<Decoder> make_decoder(const std::string&      type,
                                             const FlashBP<LoggerT>& bp,
                                             LoggerT                 logger,
                                             int                     tensor_degree,
                                             int                     bond_dim)
{
    if (type == "simple")
        return std::make_unique<SimpleDecoder<LoggerT>>(bp, std::move(logger));
    if (type == "tensor")
        return std::make_unique<TensorDecoder<LoggerT>>(bp, std::move(logger),
                                                       tensor_degree);
    if (type == "fvs")
        return std::make_unique<FvsDecoder<LoggerT>>(bp, std::move(logger),
                                                    tensor_degree);
    if (type == "degree")
        return std::make_unique<DegreeDecoder<LoggerT>>(bp, std::move(logger),
                                                       tensor_degree);
    if (type == "ml" || type == "maximum_likelihood")
        return std::make_unique<MaximumLikelihoodDecoder<LoggerT>>(
            bp, std::move(logger), bond_dim);

    throw std::invalid_argument("Unknown decoder type: \"" + type + "\". "
                                "Available: \"simple\", \"tensor\", \"fvs\", "
                                "\"degree\", \"ml\".");
}

// ── FlashBP<LoggerT> constructor ──────────────────────────────────────────────

template<typename LoggerT>
FlashBP<LoggerT>::FlashBP(py::object dem, py::object config)
    : logger_(make_logger<LoggerT>(config))
{
    num_detectors   = py::cast<int>(dem.attr("num_detectors"));
    num_observables = py::cast<int>(dem.attr("num_observables"));

    py::object flat = dem.attr("flattened")();

    num_errors = 0;
    for (py::handle instr : flat)
        if (py::cast<std::string>(instr.attr("type")) == "error")
            ++num_errors;

    H_.assign((size_t)num_detectors  * num_errors, 0);
    L_.assign((size_t)num_observables * num_errors, 0);
    error_probs_.resize(num_errors, 0.0);

    int e = 0;
    for (py::handle instr : flat) {
        if (py::cast<std::string>(instr.attr("type")) != "error") continue;

        py::list args   = instr.attr("args_copy")();
        error_probs_[e] = py::cast<double>(args[0]);

        py::list targets = instr.attr("targets_copy")();
        for (py::handle t : targets) {
            if (py::cast<bool>(t.attr("is_separator")())) continue;
            int val = py::cast<int>(t.attr("val"));
            if (py::cast<bool>(t.attr("is_relative_detector_id")()))
                H_[(size_t)val * num_errors + e] = 1;
            else if (py::cast<bool>(t.attr("is_logical_observable_id")()))
                L_[(size_t)val * num_errors + e] = 1;
        }
        ++e;
    }

    logger_("FlashBP constructed  detectors=" + std::to_string(num_detectors) +
            "  errors=" + std::to_string(num_errors), 1);

    decoder_ = make_decoder(
        py::cast<std::string>(config.attr("decoder")),
        *this, logger_,
        py::cast<int>(config.attr("degree")),
        py::cast<int>(config.attr("bond_dim")));
}

template<typename LoggerT>
FlashBP<LoggerT>::~FlashBP() = default;

// ── decode ────────────────────────────────────────────────────────────────────

template<typename LoggerT>
py::array_t<uint8_t> FlashBP<LoggerT>::decode(py::array_t<uint8_t> syndrome_arr,
                                               int max_iter) const
{
    auto s = syndrome_arr.unchecked<1>();
    if ((int)s.shape(0) != num_detectors)
        throw std::invalid_argument("Syndrome length must equal num_detectors.");

    std::vector<uint8_t> syn(num_detectors);
    for (int i = 0; i < num_detectors; ++i) syn[i] = s(i);

    auto result = (*decoder_)(syn, max_iter);

    py::array_t<uint8_t> out((py::ssize_t)result.size());
    auto buf = out.mutable_unchecked<1>();
    for (size_t i = 0; i < result.size(); ++i) buf(i) = result[i];
    return out;
}

// ── matrix accessors ──────────────────────────────────────────────────────────

template<typename LoggerT>
py::array_t<uint8_t> FlashBP<LoggerT>::get_H() const {
    py::array_t<uint8_t> out({num_detectors, num_errors});
    auto buf = out.mutable_unchecked<2>();
    for (int d = 0; d < num_detectors; ++d)
        for (int er = 0; er < num_errors; ++er)
            buf(d, er) = H_[(size_t)d * num_errors + er];
    return out;
}

template<typename LoggerT>
py::array_t<uint8_t> FlashBP<LoggerT>::get_L() const {
    py::array_t<uint8_t> out({num_observables, num_errors});
    auto buf = out.mutable_unchecked<2>();
    for (int o = 0; o < num_observables; ++o)
        for (int er = 0; er < num_errors; ++er)
            buf(o, er) = L_[(size_t)o * num_errors + er];
    return out;
}

template<typename LoggerT>
py::array_t<double> FlashBP<LoggerT>::get_error_probs() const {
    py::array_t<double> out(num_errors);
    auto buf = out.mutable_unchecked<1>();
    for (int er = 0; er < num_errors; ++er) buf(er) = error_probs_[er];
    return out;
}

// ── get_recording — default no-op; specialised for RecordLogger ───────────────

namespace {
template<typename T>
py::array_t<T> vec_to_numpy(const std::vector<T>& v) {
    py::array_t<T> arr((py::ssize_t)v.size());
    std::copy(v.begin(), v.end(), arr.mutable_data());
    return arr;
}
}

namespace {
py::object shots_to_py(const std::vector<ShotRecord>& shots) {
    py::list out;
    for (const auto& sr : shots) {
        py::list iters;
        for (const auto& ir : sr.iterations) {
            py::dict it;
            it["iteration"] = ir.iteration;
            it["syndrome"]  = vec_to_numpy(ir.syndrome);
            it["decision"]  = vec_to_numpy(ir.decision);
            it["msg_v2c"]   = vec_to_numpy(ir.msg_v2c);
            it["msg_c2v"]   = vec_to_numpy(ir.msg_c2v);
            if (!ir.tensors.empty()) {
                py::list tensors;
                for (const auto& ct : ir.tensors) {
                    py::dict t;
                    t["check_idx"] = ct.check_idx;
                    t["nbhd_data"] = vec_to_numpy(ct.nbhd_data);
                    t["weight"]    = vec_to_numpy(ct.weight);
                    t["parity"]    = vec_to_numpy(ct.parity);
                    tensors.append(t);
                }
                it["tensors"] = tensors;
            }
            iters.append(it);
        }
        py::dict shot_dict;
        shot_dict["batch"]      = sr.batch;
        shot_dict["shot"]       = sr.shot;
        shot_dict["iterations"] = iters;
        out.append(shot_dict);
    }
    return out;
}

py::object ml_shots_to_py(const std::vector<MLShotRecord>& shots) {
    py::list out;
    for (const auto& sr : shots) {
        py::list steps;
        for (const auto& step : sr.steps) {
            py::dict st;
            st["axis"]        = step.axis;
            st["error_idx"]   = step.error_idx;
            st["duration_us"] = step.duration_us;
            st["state_bits"]  = step.state_bits;
            st["num_states"]  = step.num_states;
            st["states"]      = vec_to_numpy(step.states);
            st["log_probs"]   = vec_to_numpy(step.log_probs);
            st["class_log_probs"] = vec_to_numpy(step.class_log_probs);
            steps.append(st);
        }
        py::dict shot;
        shot["batch"]           = sr.batch;
        shot["shot"]            = sr.shot;
        shot["syndrome"]        = vec_to_numpy(sr.syndrome);
        shot["num_detectors"]   = sr.num_detectors;
        shot["num_observables"] = sr.num_observables;
        shot["device"]          = sr.device;
        shot["steps"]           = steps;
        shot["class_log_probs"] = vec_to_numpy(sr.class_log_probs);
        shot["best_class"]      = sr.best_class;
        out.append(shot);
    }
    return out;
}
} // anonymous

template<typename LoggerT>
py::object FlashBP<LoggerT>::get_recording() const { return py::none(); }

template<>
py::object FlashBP<RecordLogger>::get_recording() const {
    return shots_to_py(logger_.shots());
}

template<>
py::object FlashBP<TensorLogger>::get_recording() const {
    return shots_to_py(logger_.shots());
}

template<>
py::object FlashBP<MLLogger>::get_recording() const {
    return ml_shots_to_py(logger_.ml_shots());
}

// ── Explicit instantiations ───────────────────────────────────────────────────

template class FlashBP<Logger<false>>;
template class FlashBP<Logger<true>>;
template class FlashBP<DecodeLogger<true>>;
template class FlashBP<RecordLogger>;
template class FlashBP<TensorLogger>;
template class FlashBP<MLLogger>;

// ── Public factory ────────────────────────────────────────────────────────────

std::unique_ptr<FlashBPBase> make_flashbp(py::object dem, py::object config) {
    if (!py::cast<bool>(config.attr("log")))
        return std::make_unique<FlashBP<Logger<false>>>(dem, config);

    const std::string log_type = py::cast<std::string>(config.attr("log_type"));
    if (log_type == "decode")
        return std::make_unique<FlashBP<DecodeLogger<true>>>(dem, config);
    if (log_type == "record")
        return std::make_unique<FlashBP<RecordLogger>>(dem, config);
    if (log_type == "tensor")
        return std::make_unique<FlashBP<TensorLogger>>(dem, config);
    if (log_type == "ml") {
        const std::string decoder = py::cast<std::string>(config.attr("decoder"));
        if (decoder != "ml" && decoder != "maximum_likelihood")
            throw std::invalid_argument("log_type='ml' requires decoder='ml'.");
        return std::make_unique<FlashBP<MLLogger>>(dem, config);
    }
    return std::make_unique<FlashBP<Logger<true>>>(dem, config);
}
