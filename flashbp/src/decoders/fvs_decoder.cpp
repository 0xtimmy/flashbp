#include "fvs_decoder.h"
#include "../flashbp_core.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace {

inline double minsum_cost(int x, double l) {
    return std::max(0.0, (x ? 1.0 : -1.0) * l);
}

inline int popcount32(uint32_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(v);
#else
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return (int)((((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
#endif
}

} // anonymous namespace

// ── Constructor ───────────────────────────────────────────────────────────────

template<typename LoggerT>
FvsDecoder<LoggerT>::FvsDecoder(const FlashBPBase& bp,
                                LoggerT            logger,
                                int                degree)
    : num_detectors_(bp.num_detectors)
    , num_errors_(bp.num_errors)
    , degree_(degree)
    , logger_(std::move(logger))
{
    if (degree < 1)
        throw std::invalid_argument("FvsDecoder: degree must be >= 1.");

    const auto& H           = bp.H_raw();
    const auto& error_probs = bp.error_probs_raw();

    // ── Tanner edges + adjacency ─────────────────────────────────────────────
    var_edges_.assign(num_errors_,     {});
    check_edges_.assign(num_detectors_, {});
    for (int d = 0; d < num_detectors_; ++d)
        for (int e = 0; e < num_errors_; ++e)
            if (H[(size_t)d * num_errors_ + e]) {
                int idx = (int)edges_.size();
                edges_.push_back({d, e});
                var_edges_[e].push_back(idx);
                check_edges_[d].push_back(idx);
            }

    ch_llr_.resize(num_errors_);
    for (int e = 0; e < num_errors_; ++e) {
        double p = std::max(1e-12, std::min(1.0 - 1e-12, error_probs[e]));
        ch_llr_[e] = std::log((1.0 - p) / p);
    }

    // ── Per-check: N_d(c), cycles, FVS, partitioned constraints ──────────────
    nbhd_data_       .assign(num_detectors_, {});
    fvs_             .assign(num_detectors_, {});
    residual_        .assign(num_detectors_, {});
    f_only_checks_   .assign(num_detectors_, {});
    residual_graph_  .assign(num_detectors_, {});
    fvs_axis_edge_   .assign(num_detectors_, {});
    edge_fvs_pos_    .assign(edges_.size(), -1);

    int max_F = 0, max_R = 0, max_R_edges = 0;

    for (int c = 0; c < num_detectors_; ++c) {
        // ----- 1. N_d(c) via alternating BFS -----------------------------
        std::set<int>    data_in_nbhd;
        std::set<int>    visited_checks{c};
        std::vector<int> frontier_checks{c};
        for (int hop = 0; hop < degree; ++hop) {
            std::vector<int> new_data;
            for (int cc : frontier_checks)
                for (int ei : check_edges_[cc]) {
                    int v = edges_[ei].var;
                    if (data_in_nbhd.insert(v).second) new_data.push_back(v);
                }
            if (hop + 1 == degree) break;
            std::vector<int> new_checks;
            for (int v : new_data)
                for (int ei : var_edges_[v]) {
                    int cc = edges_[ei].check;
                    if (visited_checks.insert(cc).second) new_checks.push_back(cc);
                }
            if (new_checks.empty()) break;
            frontier_checks = std::move(new_checks);
        }
        nbhd_data_[c].assign(data_in_nbhd.begin(), data_in_nbhd.end());
        const int K = (int)nbhd_data_[c].size();

        std::vector<int> axis_of_v(num_errors_, -1);
        for (int k = 0; k < K; ++k) axis_of_v[nbhd_data_[c][k]] = k;

        // ----- 2. Internal checks (N(c') ⊆ N_d(c)) -----------------------
        std::unordered_set<int> candidate_checks;
        for (int v : nbhd_data_[c])
            for (int ei : var_edges_[v])
                candidate_checks.insert(edges_[ei].check);

        struct PartialCheck { int check_idx; std::vector<int> axes_in_nbhd; };
        std::vector<PartialCheck> internal_partial;
        for (int cc : candidate_checks) {
            std::vector<int> axes;
            bool fully_in = true;
            for (int ei : check_edges_[cc]) {
                int k = axis_of_v[edges_[ei].var];
                if (k < 0) { fully_in = false; break; }
                axes.push_back(k);
            }
            if (fully_in) internal_partial.push_back({cc, std::move(axes)});
        }

        // ----- 3. Subgraph adjacency for cycle search --------------------
        const int n_int   = (int)internal_partial.size();
        const int n_total = K + n_int;
        std::vector<std::vector<int>> sub_adj(n_total);
        for (int ci = 0; ci < n_int; ++ci) {
            int check_node = K + ci;
            for (int k : internal_partial[ci].axes_in_nbhd) {
                sub_adj[k].push_back(check_node);
                sub_adj[check_node].push_back(k);
            }
        }

        // ----- 4-5. Complete FVS: seed with N(c), break every remaining
        //            cycle in S_d(c) via iterative max-degree cycle-removal.
        //
        // We need a *complete* FVS (not just one covering short cycles): the
        // residual will be fed to min-sum BP which is only exact when the
        // residual graph is acyclic.  Stopping at "cycles up to 2*degree"
        // leaves girth-> 2*degree codes (e.g. Gross) with a loopy residual,
        // which corrupts inner BP and degrades decoding.
        std::vector<bool> in_fvs(K, false);
        std::vector<int>  fvs_positions;
        for (int ei : check_edges_[c]) {
            int k = axis_of_v[edges_[ei].var];
            if (k >= 0 && !in_fvs[k]) {
                in_fvs[k] = true;
                fvs_positions.push_back(k);
            }
        }
        {
            std::vector<bool> removed(n_total, false);
            for (int k = 0; k < K; ++k) if (in_fvs[k]) removed[k] = true;

            std::vector<int>  state(n_total, 0);   // 0 = unseen, 1 = on stack, 2 = done
            std::vector<int>  parent_arr(n_total, -1);
            std::vector<int>  cycle_buf;

            std::function<bool(int, int)> dfs = [&](int u, int par) -> bool {
                state[u] = 1;
                parent_arr[u] = par;
                for (int v : sub_adj[u]) {
                    if (removed[v]) continue;
                    if (state[v] == 1 && v != par) {
                        // Back-edge u -> v: cycle = v, u, parent(u), ..., until v
                        cycle_buf.clear();
                        cycle_buf.push_back(v);
                        int cur = u;
                        while (cur != v) {
                            cycle_buf.push_back(cur);
                            cur = parent_arr[cur];
                            if (cur == -1) { cycle_buf.clear(); return false; }
                        }
                        return true;
                    }
                    if (state[v] == 0 && dfs(v, u)) return true;
                }
                state[u] = 2;
                return false;
            };

            while (true) {
                std::fill(state.begin(), state.end(), 0);
                std::fill(parent_arr.begin(), parent_arr.end(), -1);
                bool found = false;
                for (int start = 0; start < n_total && !found; ++start) {
                    if (removed[start] || state[start] != 0) continue;
                    if (dfs(start, -1)) found = true;
                }
                if (!found) break;

                // Pick highest-degree variable on the cycle to remove.
                int best_var = -1, best_deg = -1;
                for (int n : cycle_buf) {
                    if (n >= K) continue;  // bipartite — only remove vars
                    int deg = 0;
                    for (int nb : sub_adj[n]) if (!removed[nb]) ++deg;
                    if (deg > best_deg) { best_deg = deg; best_var = n; }
                }
                if (best_var < 0) break;
                in_fvs[best_var] = true;
                removed[best_var] = true;
                fvs_positions.push_back(best_var);
            }
        }

        // ----- 6. Partition axes (F sorted by axis order, R the rest) ----
        std::sort(fvs_positions.begin(), fvs_positions.end());
        std::vector<int> resid_positions;
        for (int k = 0; k < K; ++k)
            if (!in_fvs[k]) resid_positions.push_back(k);

        const int K_F = (int)fvs_positions.size();
        const int K_R = (int)resid_positions.size();
        max_F = std::max(max_F, K_F);
        max_R = std::max(max_R, K_R);

        if (K_F > MAX_FVS)
            throw std::runtime_error(
                "FvsDecoder: check " + std::to_string(c) +
                " requires |F|=" + std::to_string(K_F) +
                " for a complete FVS (the local girth forces this) but "
                "MAX_FVS=" + std::to_string(MAX_FVS) +
                ".  Lower `degree` or raise MAX_FVS (cost scales as 2^|F|).");

        fvs_[c].resize(K_F);
        residual_[c].resize(K_R);
        for (int k = 0; k < K_F; ++k) fvs_[c][k]      = nbhd_data_[c][fvs_positions[k]];
        for (int k = 0; k < K_R; ++k) residual_[c][k] = nbhd_data_[c][resid_positions[k]];

        std::vector<int> f_pos_of_axis(K, -1), r_pos_of_axis(K, -1);
        for (int k = 0; k < K_F; ++k) f_pos_of_axis[fvs_positions[k]] = k;
        for (int k = 0; k < K_R; ++k) r_pos_of_axis[resid_positions[k]] = k;

        // ----- 7. Partition internal checks into F-only and residual -----
        ResidualGraph& rg = residual_graph_[c];
        rg.axis_edges.assign(K_R, {});

        for (const auto& pc : internal_partial) {
            uint32_t f_mask = 0, r_mask = 0;
            std::vector<int> r_axes;
            for (int k : pc.axes_in_nbhd) {
                int fp = f_pos_of_axis[k];
                int rp = r_pos_of_axis[k];
                if (fp >= 0) {
                    f_mask |= (uint32_t)1 << fp;
                } else if (rp >= 0) {
                    r_mask |= (uint32_t)1 << rp;
                    r_axes.push_back(rp);
                }
            }

            if (r_mask == 0) {
                // Entirely inside F — checked up front per x_F
                f_only_checks_[c].push_back({f_mask, pc.check_idx});
            } else {
                // Residual check — participates in inner BP
                int rc = (int)rg.check_indices.size();
                rg.check_indices.push_back(pc.check_idx);
                rg.check_f_masks.push_back(f_mask);
                rg.check_r_masks.push_back(r_mask);

                std::vector<int> e_list;
                for (int rp : r_axes) {
                    int e = rg.n_edges++;
                    e_list.push_back(e);
                    rg.axis_edges[rp].push_back(e);
                }
                rg.check_edges.push_back(std::move(e_list));
            }
        }
        max_R_edges = std::max(max_R_edges, rg.n_edges);

        // ----- 8. Per-axis incoming-edge tables for F --------------------
        fvs_axis_edge_[c].assign(K_F, -1);
        for (int ei : check_edges_[c]) {
            int k  = axis_of_v[edges_[ei].var];
            int fp = f_pos_of_axis[k];
            fvs_axis_edge_[c][fp] = ei;
            edge_fvs_pos_[ei]     = fp;
        }
    }

    logger_("FvsDecoder constructed  degree=" + std::to_string(degree_) +
            "  edges="         + std::to_string(edges_.size()) +
            "  max_|F|="       + std::to_string(max_F) +
            "  max_|R|="       + std::to_string(max_R) +
            "  max_R_edges="   + std::to_string(max_R_edges), 1);
}

// ── operator() ────────────────────────────────────────────────────────────────

template<typename LoggerT>
std::vector<uint8_t> FvsDecoder<LoggerT>::operator()(
    const std::vector<uint8_t>& syndrome,
    int max_iter)
{
    if ((int)syndrome.size() != num_detectors_)
        throw std::invalid_argument("Syndrome length must equal num_detectors.");

    if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>) {
        logger_.set_shot(shot_counter_);
        logger_.set_iteration(0);
    }
    ++shot_counter_;

    logger_("decode called  syndrome_weight=" +
            std::to_string([&]{ int w=0; for (auto b:syndrome) w+=b; return w; }()),
            3);

    const int num_edges = (int)edges_.size();
    std::vector<double> msg_v2c(num_edges);
    std::vector<double> msg_c2v(num_edges, 0.0);
    for (int i = 0; i < num_edges; ++i) msg_v2c[i] = ch_llr_[edges_[i].var];

    std::vector<uint8_t> decision(num_errors_);

    constexpr double INF = std::numeric_limits<double>::infinity();
    constexpr double SAT = 1e30;

    // Reusable inner-BP buffers (resized per check as needed)
    std::vector<double>  rmsg_v2c, rmsg_c2v;
    std::vector<int>     hard_dec;

    for (int iter = 0; iter < max_iter; ++iter) {
        if constexpr (std::is_base_of_v<DecodeLogger<true>, LoggerT>)
            logger_.set_iteration((unsigned int)iter);
        logger_("iteration=" + std::to_string(iter), 3);

        // ── check-node update ─────────────────────────────────────────────
        for (int c = 0; c < num_detectors_; ++c) {
            const int K_F = (int)fvs_[c].size();
            const int K_R = (int)residual_[c].size();
            const size_t N_F = (size_t)1 << K_F;

            const ResidualGraph& rg = residual_graph_[c];
            const int n_rc   = (int)rg.check_indices.size();
            const int n_redg = rg.n_edges;

            // Gather incoming LLR per F axis
            std::vector<double> l_F(K_F);
            for (int k = 0; k < K_F; ++k) {
                int ei = fvs_axis_edge_[c][k];
                l_F[k] = (ei >= 0) ? msg_v2c[ei] : ch_llr_[fvs_[c][k]];
            }
            // R axes are deep — channel prior only
            std::vector<double> l_R(K_R);
            for (int k = 0; k < K_R; ++k) l_R[k] = ch_llr_[residual_[c][k]];

            const int n_out = (int)check_edges_[c].size();
            std::vector<double> W0(n_out, INF), W1(n_out, INF);

            // Allocate inner-BP scratch
            if ((int)rmsg_v2c.size() < n_redg) {
                rmsg_v2c.resize(n_redg);
                rmsg_c2v.resize(n_redg);
            }
            hard_dec.assign(K_R, 0);

            for (size_t x_F = 0; x_F < N_F; ++x_F) {
                // ---- F-only constraint pruning -------------------------
                bool f_only_violated = false;
                for (const auto& fc : f_only_checks_[c]) {
                    int s   = syndrome[fc.check_idx] & 1;
                    int par = popcount32((uint32_t)x_F & fc.f_mask) ^ s;
                    if (par & 1) { f_only_violated = true; break; }
                }
                if (f_only_violated) continue;

                // ---- F-cost --------------------------------------------
                double cost_F = 0.0;
                for (int k = 0; k < K_F; ++k)
                    cost_F += minsum_cost((int)((x_F >> k) & 1), l_F[k]);

                // ---- inner min-sum BP on the residual subgraph ---------
                double cost_R = 0.0;
                bool   feasible = true;

                if (n_redg == 0) {
                    // No residual checks coupled to R; only the channel
                    // priors determine the residual cost.  Pick the better
                    // value at each R axis.
                    for (int k = 0; k < K_R; ++k) {
                        hard_dec[k] = (l_R[k] < 0.0) ? 1 : 0;
                        cost_R += minsum_cost(hard_dec[k], l_R[k]);
                    }
                } else {
                    // Compute effective syndromes for each residual check
                    std::vector<uint8_t> eff_syn(n_rc);
                    for (int rc = 0; rc < n_rc; ++rc) {
                        int s     = syndrome[rg.check_indices[rc]] & 1;
                        int f_par = popcount32((uint32_t)x_F & rg.check_f_masks[rc]) & 1;
                        eff_syn[rc] = (uint8_t)(s ^ f_par);
                    }

                    // Initialise messages: msg_v2c = channel prior on each axis
                    for (int k = 0; k < K_R; ++k)
                        for (int e : rg.axis_edges[k])
                            rmsg_v2c[e] = l_R[k];
                    for (int e = 0; e < n_redg; ++e) rmsg_c2v[e] = 0.0;

                    // Min-sum sweeps
                    for (int it = 0; it < INNER_BP_ITERS; ++it) {
                        // Check update
                        for (int rc = 0; rc < n_rc; ++rc) {
                            const auto& es = rg.check_edges[rc];
                            int deg = (int)es.size();
                            if (deg == 0) continue;

                            int sign_prod = (eff_syn[rc] == 1) ? -1 : 1;
                            for (int e : es)
                                sign_prod *= (rmsg_v2c[e] >= 0.0) ? 1 : -1;

                            double min1 = INF, min2 = INF;
                            int argmin1 = 0;
                            for (int i = 0; i < deg; ++i) {
                                double m = std::abs(rmsg_v2c[es[i]]);
                                if (m < min1) { min2 = min1; min1 = m; argmin1 = i; }
                                else if (m < min2) { min2 = m; }
                            }
                            for (int i = 0; i < deg; ++i) {
                                int e = es[i];
                                int v_sign = (rmsg_v2c[e] >= 0.0) ? 1 : -1;
                                int out_sign = sign_prod * v_sign;
                                double out_mag = (i == argmin1) ? min2 : min1;
                                rmsg_c2v[e] = (double)out_sign * out_mag;
                            }
                        }

                        // Variable update
                        for (int k = 0; k < K_R; ++k) {
                            double total = l_R[k];
                            for (int e : rg.axis_edges[k]) total += rmsg_c2v[e];
                            for (int e : rg.axis_edges[k]) rmsg_v2c[e] = total - rmsg_c2v[e];
                        }
                    }

                    // Hard decisions
                    for (int k = 0; k < K_R; ++k) {
                        double total = l_R[k];
                        for (int e : rg.axis_edges[k]) total += rmsg_c2v[e];
                        hard_dec[k] = (total < 0.0) ? 1 : 0;
                    }

                    // Feasibility: every residual check must be satisfied
                    uint32_t x_R_bits = 0;
                    for (int k = 0; k < K_R; ++k)
                        if (hard_dec[k]) x_R_bits |= (uint32_t)1 << k;
                    for (int rc = 0; rc < n_rc; ++rc) {
                        int s     = eff_syn[rc];
                        int r_par = popcount32(x_R_bits & rg.check_r_masks[rc]) & 1;
                        if ((s ^ r_par) & 1) { feasible = false; break; }
                    }

                    if (feasible)
                        for (int k = 0; k < K_R; ++k)
                            cost_R += minsum_cost(hard_dec[k], l_R[k]);
                }

                if (!feasible) continue;

                const double total = cost_F + cost_R;

                // Marginalise into per-outgoing-edge W_0, W_1
                for (int oei = 0; oei < n_out; ++oei) {
                    int ei  = check_edges_[c][oei];
                    int fp  = edge_fvs_pos_[ei];
                    int x_v = (int)((x_F >> fp) & 1);
                    double w_ext = total - minsum_cost(x_v, l_F[fp]);
                    if (x_v == 0) { if (w_ext < W0[oei]) W0[oei] = w_ext; }
                    else          { if (w_ext < W1[oei]) W1[oei] = w_ext; }
                }
            }

            // Write outgoing messages
            for (int oei = 0; oei < n_out; ++oei) {
                int ei = check_edges_[c][oei];
                if      (W0[oei] == INF && W1[oei] == INF) msg_c2v[ei] =  0.0;
                else if (W0[oei] == INF)                   msg_c2v[ei] = -SAT;
                else if (W1[oei] == INF)                   msg_c2v[ei] = +SAT;
                else                                       msg_c2v[ei] = W1[oei] - W0[oei];
            }
        }

        // ── variable-node update + hard decision ─────────────────────────
        for (int e = 0; e < num_errors_; ++e) {
            double total = ch_llr_[e];
            for (int i : var_edges_[e]) total += msg_c2v[i];
            decision[e] = (total < 0.0) ? 1u : 0u;
            for (int i : var_edges_[e]) msg_v2c[i] = total - msg_c2v[i];
        }

        if constexpr (std::is_base_of_v<RecordLogger, LoggerT>)
            logger_.record_iteration(iter, syndrome, decision, msg_v2c, msg_c2v);

        // ── convergence check ────────────────────────────────────────────
        bool converged = true;
        for (int d = 0; d < num_detectors_ && converged; ++d) {
            int p = 0;
            for (int ei : check_edges_[d]) p ^= decision[edges_[ei].var];
            if (p != (int)syndrome[d]) converged = false;
        }
        if (converged) {
            logger_("converged at iter=" + std::to_string(iter), 3);
            return decision;
        }
    }

    logger_("max_iter=" + std::to_string(max_iter) +
            " reached without convergence", 2);
    return decision;
}

// ── Explicit instantiations ───────────────────────────────────────────────────
template class FvsDecoder<Logger<false>>;
template class FvsDecoder<Logger<true>>;
template class FvsDecoder<DecodeLogger<true>>;
template class FvsDecoder<RecordLogger>;
template class FvsDecoder<TensorLogger>;
template class FvsDecoder<MLLogger>;
