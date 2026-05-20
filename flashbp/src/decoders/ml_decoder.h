#pragma once
#include "decoder.h"
#include "../loggers/tensor_logger.h"
#include "../loggers/ml_logger.h"
#include <cstdint>
#include <vector>

class FlashBPBase;

// ---------------------------------------------------------------------------
// MaximumLikelihoodDecoder — coset-marginal (a.k.a. ML class) decoder.
//
// Given a syndrome s, the optimal CSS decoder doesn't pick the single most
// likely error e — it picks the most likely *logical equivalence class*
// L·e mod 2, summing the probability of every syndrome-consistent error
// inside that class:
//
//      P(class L | s)  =  Σ_{e : H·e = s, L·e = ℓ}  P(e)
//
// Decoding returns any e in the argmax class.  This is the ground-truth
// reference every other decoder in the package is approximating.
//
// SCAFFOLD STATUS
// ---------------
// The eventual implementation is a tensor-network contraction with a
// `bond_dim` (χ) hyperparameter controlling the accuracy/cost trade-off
// (matches the "complexity scales with entanglement" intuition you wanted
// from the start).  This scaffold instead does an **exact brute-force**
// enumeration of all 2^num_errors error vectors per shot, accumulating
// log-probabilities into 2^num_observables logical-class buckets via
// log-sum-exp.  Correct, intractable beyond ~20 errors.
//
// The brute-force path is wrapped in a small helper so the upcoming TN
// implementation just swaps out `class_log_probs(syndrome)`.
// ---------------------------------------------------------------------------
template<typename LoggerT>
class MaximumLikelihoodDecoder : public Decoder {
public:
    static constexpr int MAX_BRUTE_FORCE_BITS = 22;   // 4 M errors per shot

    explicit MaximumLikelihoodDecoder(const FlashBPBase& bp,
                                      LoggerT            logger,
                                      int                bond_dim);

    std::vector<uint8_t> operator()(
        const std::vector<uint8_t>& syndrome,
        int max_iter = 1
    ) override;

    int num_detectors() const override { return num_detectors_; }
    int num_errors()    const override { return num_errors_; }

private:
    // Compute log P(class ℓ | syndrome) for every logical class ℓ.
    // Returns a vector of length 2^num_observables.  This is the function
    // the eventual TN implementation will replace.
    std::vector<double> class_log_probs(const std::vector<uint8_t>& syndrome) const;

    // Exact Torch-backed contraction over accumulated (syndrome, logical)
    // states. Used when LibTorch is linked and the dense state vector is
    // tractable.
    std::vector<double> class_log_probs_torch(
        const std::vector<uint8_t>& syndrome) const;

    // Detect disconnected Tanner components and solve their ML logical
    // distributions separately. Returns true when a split was used.
    bool class_log_probs_split(
        const std::vector<uint8_t>& syndrome,
        std::vector<double>&       out_logp) const;

    // Find a syndrome-consistent error in a specified logical class
    // (currently the highest-probability single error in that class —
    // discovered as a by-product of the brute-force enumeration above).
    std::vector<uint8_t> find_representative(
        const std::vector<uint8_t>& syndrome,
        int target_class) const;

    int num_detectors_;
    int num_errors_;
    int num_observables_;
    int bond_dim_;

    // Stored from the FlashBP host so we don't keep a pointer back.
    std::vector<uint8_t> H_;             // num_detectors x num_errors, row-major
    std::vector<uint8_t> L_;             // num_observables x num_errors, row-major
    std::vector<double>  log_p_;         // log(p_e)
    std::vector<double>  log_1mp_;       // log(1 - p_e)

    LoggerT      logger_;
    unsigned int shot_counter_ = 0;
};

extern template class MaximumLikelihoodDecoder<Logger<false>>;
extern template class MaximumLikelihoodDecoder<Logger<true>>;
extern template class MaximumLikelihoodDecoder<DecodeLogger<true>>;
extern template class MaximumLikelihoodDecoder<RecordLogger>;
extern template class MaximumLikelihoodDecoder<TensorLogger>;
extern template class MaximumLikelihoodDecoder<MLLogger>;
