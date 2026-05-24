#include "flashbp_core.h"
#include "decoders/decoder.h"
#include "decoders/simple_decoder.h"
#include "decoders/tensor_decoder.h"
#include "decoders/gbp_decoder.h"
#include "decoders/degree_decoder.h"
#include "decoders/ml_decoder.h"
#include "decoders/surprise_ml_decoder.h"
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
GBPLogger make_logger<GBPLogger>(py::object config) {
    const unsigned int level    = py::cast<unsigned int>(config.attr("log_level"));
    const bool         console  = py::cast<bool>(config.attr("log_console"));
    const bool         buffered = py::cast<bool>(config.attr("log_buffered"));
    const std::string  file     = config.attr("log_file").is_none()
                                   ? std::string{}
                                   : py::cast<std::string>(config.attr("log_file"));
    return GBPLogger{level, console, buffered, file};
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

template<>
SurpriseMLLogger make_logger<SurpriseMLLogger>(py::object config) {
    const unsigned int level    = py::cast<unsigned int>(config.attr("log_level"));
    const bool         console  = py::cast<bool>(config.attr("log_console"));
    const bool         buffered = py::cast<bool>(config.attr("log_buffered"));
    const std::string  file     = config.attr("log_file").is_none()
                                   ? std::string{}
                                   : py::cast<std::string>(config.attr("log_file"));
    return SurpriseMLLogger{level, console, buffered, file};
}

// ── Internal decoder factory ──────────────────────────────────────────────────

template<typename LoggerT>
static std::unique_ptr<Decoder> make_decoder(const std::string&      type,
                                             const FlashBP<LoggerT>& bp,
                                             LoggerT                 logger,
                                             int                     tensor_degree,
                                             int                     bond_dim,
                                             const std::string&      region_policy,
                                             const std::string&      gbp_backend,
                                             uint64_t                gbp_max_states,
                                             std::vector<GBPManualGroup> gbp_manual_groups,
                                             bool                    gbp_manual_add_single_checks,
                                             double                  gbp_oscillation_boost,
                                             double                  gbp_oscillation_boost_cap)
{
    auto gbp_budget = [&]() {
        const bool sparse_budget =
            gbp_backend == "sparse" || gbp_backend == "sparse_cpu";
        GBPRegionBudget budget;
        budget.max_axes = sparse_budget
                          ? GBPDecoder<LoggerT>::MAX_SPARSE_AXES
                          : GBPDecoder<LoggerT>::MAX_AXES;
        budget.max_states = gbp_max_states
                            ? gbp_max_states
                            : GBPDecoder<LoggerT>::DEFAULT_MAX_STATES;
        budget.use_valid_state_estimate = sparse_budget;
        return budget;
    };

    if constexpr (std::is_same_v<LoggerT, SurpriseMLLogger>) {
        if (type == "surprise_ml" || type == "surprise_maximum_likelihood")
            return std::make_unique<SurpriseMLDecoder>(
                bp, std::move(logger), bond_dim);
        throw std::invalid_argument(
            "log_type='surprise_ml' only supports decoder='surprise_ml'.");
    } else if constexpr (std::is_same_v<LoggerT, GBPLogger>) {
        if (type == "gbp" || type == "generalized_belief_propagation")
            return std::make_unique<GBPDecoder<LoggerT>>(
                bp,
                std::move(logger),
                tensor_degree,
                make_region_grouping_policy(region_policy, tensor_degree,
                                            gbp_budget(), gbp_manual_groups,
                                            gbp_manual_add_single_checks),
                make_gbp_backend(gbp_backend),
                gbp_oscillation_boost,
                gbp_oscillation_boost_cap);
        throw std::invalid_argument("log_type='gbp' only supports decoder='gbp'.");
    } else {
    if (type == "simple")
        return std::make_unique<SimpleDecoder<LoggerT>>(bp, std::move(logger));
    if (type == "tensor")
        return std::make_unique<TensorDecoder<LoggerT>>(bp, std::move(logger),
                                                       tensor_degree);
    if (type == "gbp" || type == "generalized_belief_propagation")
        return std::make_unique<GBPDecoder<LoggerT>>(
            bp,
            std::move(logger),
            tensor_degree,
            make_region_grouping_policy(region_policy, tensor_degree,
                                        gbp_budget(), gbp_manual_groups,
                                        gbp_manual_add_single_checks),
            make_gbp_backend(gbp_backend),
            gbp_oscillation_boost,
            gbp_oscillation_boost_cap);
    if (type == "degree")
        return std::make_unique<DegreeDecoder<LoggerT>>(bp, std::move(logger),
                                                       tensor_degree);
    if (type == "ml" || type == "maximum_likelihood")
        return std::make_unique<MaximumLikelihoodDecoder<LoggerT>>(
            bp, std::move(logger), bond_dim);

    throw std::invalid_argument("Unknown decoder type: \"" + type + "\". "
                                "Available: \"simple\", \"tensor\", \"gbp\", "
                                "\"degree\", \"ml\", \"surprise_ml\".");
    }
}

// ── FlashBP<LoggerT> constructor ──────────────────────────────────────────────

namespace {
std::vector<GBPManualGroup> parse_gbp_manual_groups(py::object config);
}

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
        py::cast<int>(config.attr("bond_dim")),
        py::cast<std::string>(config.attr("region_policy")),
        py::cast<std::string>(config.attr("gbp_backend")),
        py::cast<uint64_t>(config.attr("gbp_max_states")),
        parse_gbp_manual_groups(config),
        py::cast<bool>(config.attr("gbp_manual_add_single_checks")),
        py::cast<double>(config.attr("gbp_oscillation_boost")),
        py::cast<double>(config.attr("gbp_oscillation_boost_cap")));
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

template<typename LoggerT>
py::dict FlashBP<LoggerT>::last_decode_stats() const {
    py::dict stats;
    stats["converged"] = decoder_ ? decoder_->last_converged() : false;
    stats["iterations"] = decoder_ ? decoder_->last_iterations() : 0;
    return stats;
}

namespace {

GBPRegionActivation parse_gbp_activation(const std::string& text) {
    if (text.empty() || text == "always")
        return GBPRegionActivation::Always;
    if (text == "any" || text == "any_active" || text == "any_on")
        return GBPRegionActivation::AnyCheckActive;
    if (text == "all" || text == "all_active" || text == "all_on")
        return GBPRegionActivation::AllChecksActive;
    throw std::invalid_argument(
        "Unknown manual GBP group activation: \"" + text +
        "\". Available: \"always\", \"any\", \"all\".");
}

std::vector<int> int_list_from_object(py::handle obj, const char* field) {
    std::vector<int> out;
    for (py::handle item : obj)
        out.push_back(py::cast<int>(item));
    if (out.empty())
        throw std::invalid_argument(
            std::string("Manual GBP group field \"") + field +
            "\" must not be empty.");
    return out;
}

std::vector<GBPManualGroup> parse_gbp_manual_groups(py::object config) {
    if (!py::hasattr(config, "gbp_manual_groups"))
        return {};
    py::object groups_obj = config.attr("gbp_manual_groups");
    if (groups_obj.is_none())
        return {};

    std::vector<GBPManualGroup> groups;
    for (py::handle item : groups_obj) {
        py::dict group_dict = py::reinterpret_borrow<py::dict>(item);
        GBPManualGroup group;
        if (!group_dict.contains("data"))
            throw std::invalid_argument(
                "Manual GBP groups require a \"data\" field.");
        if (group_dict.contains("checks"))
            group.checks = int_list_from_object(group_dict["checks"], "checks");
        else if (group_dict.contains("cycle_checks"))
            group.checks = int_list_from_object(
                group_dict["cycle_checks"], "cycle_checks");
        else
            throw std::invalid_argument(
                "Manual GBP groups require a \"checks\" or \"cycle_checks\" field.");
        group.data = int_list_from_object(group_dict["data"], "data");
        if (group_dict.contains("activation"))
            group.activation = parse_gbp_activation(
                py::cast<std::string>(group_dict["activation"]));
        if (group_dict.contains("center_check"))
            group.center_check = py::cast<int>(group_dict["center_check"]);
        groups.push_back(std::move(group));
    }
    return groups;
}

} // anonymous namespace

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
            if (!ir.active_regions.empty())
                it["active_regions"] = vec_to_numpy(ir.active_regions);
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

py::object gbp_shots_to_py(const std::vector<ShotRecord>& shots,
                           const GBPRecordingMetadata& meta,
                           bool has_meta) {
    py::object out = shots_to_py(shots);
    if (!has_meta)
        return out;

    py::list regions;
    for (int i = 0; i < (int)meta.regions.size(); ++i) {
        const auto& rec = meta.regions[i];
        py::dict r;
        r["index"] = i;
        r["center_check"] = rec.center_check;
        r["activation"] = rec.activation;
        r["data"] = vec_to_numpy(rec.data);
        r["cycle_checks"] = vec_to_numpy(rec.cycle_checks);
        r["axis_edge"] = vec_to_numpy(rec.axis_edge);
        r["output_edges"] = vec_to_numpy(rec.output_edges);
        r["output_axes"] = vec_to_numpy(rec.output_axes);
        r["internal_check_indices"] = vec_to_numpy(rec.internal_check_indices);
        r["internal_check_masks"] = vec_to_numpy(rec.internal_check_masks);
        r["dense_state_count"] = rec.dense_state_count;
        r["valid_state_count"] = rec.valid_state_count;
        regions.append(r);
    }

    py::dict gbp;
    gbp["policy"] = meta.policy;
    gbp["backend"] = meta.backend;
    gbp["degree"] = meta.degree;
    gbp["num_detectors"] = meta.num_detectors;
    gbp["num_errors"] = meta.num_errors;
    gbp["num_edges"] = meta.num_edges;
    gbp["regions"] = regions;

    py::list shots_list = py::reinterpret_borrow<py::list>(out);
    for (py::handle shot_handle : shots_list) {
        py::dict shot = py::reinterpret_borrow<py::dict>(shot_handle);
        shot["gbp"] = gbp;
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
            st["surprise_scores"] = vec_to_numpy(step.surprise_scores);
            st["kl_0_to_1"] = step.kl_0_to_1;
            st["kl_1_to_0"] = step.kl_1_to_0;
            st["js_divergence"] = step.js_divergence;
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
py::object FlashBP<GBPLogger>::get_recording() const {
    return gbp_shots_to_py(
        logger_.shots(), logger_.gbp_metadata(), logger_.has_gbp_metadata());
}

template<>
py::object FlashBP<MLLogger>::get_recording() const {
    return ml_shots_to_py(logger_.ml_shots());
}

template<>
py::object FlashBP<SurpriseMLLogger>::get_recording() const {
    return ml_shots_to_py(logger_.ml_shots());
}

// ── Explicit instantiations ───────────────────────────────────────────────────

template class FlashBP<Logger<false>>;
template class FlashBP<Logger<true>>;
template class FlashBP<DecodeLogger<true>>;
template class FlashBP<RecordLogger>;
template class FlashBP<TensorLogger>;
template class FlashBP<GBPLogger>;
template class FlashBP<MLLogger>;
template class FlashBP<SurpriseMLLogger>;

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
    if (log_type == "gbp") {
        const std::string decoder = py::cast<std::string>(config.attr("decoder"));
        if (decoder != "gbp" && decoder != "generalized_belief_propagation")
            throw std::invalid_argument("log_type='gbp' requires decoder='gbp'.");
        return std::make_unique<FlashBP<GBPLogger>>(dem, config);
    }
    if (log_type == "ml") {
        const std::string decoder = py::cast<std::string>(config.attr("decoder"));
        if (decoder != "ml" && decoder != "maximum_likelihood")
            throw std::invalid_argument("log_type='ml' requires decoder='ml'.");
        return std::make_unique<FlashBP<MLLogger>>(dem, config);
    }
    if (log_type == "surprise_ml") {
        const std::string decoder = py::cast<std::string>(config.attr("decoder"));
        if (decoder != "surprise_ml" && decoder != "surprise_maximum_likelihood")
            throw std::invalid_argument(
                "log_type='surprise_ml' requires decoder='surprise_ml'.");
        return std::make_unique<FlashBP<SurpriseMLLogger>>(dem, config);
    }
    return std::make_unique<FlashBP<Logger<true>>>(dem, config);
}
