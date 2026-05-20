"""
Animate contraction of a recorded local tensor from TensorLogger.

Each frame shows the Tanner/factor graph next to a compact chart of tensor
configurations.  The graph shades check nodes by syndrome and data nodes by
their tensor-local error likelihood.  Axes that have already been contracted
receive a heavier outline on their corresponding data nodes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .layout import bipartite_layout, edges_from_H
from .video import make_video


FAINT_EDGE = "#d0d0d0"
FAINT_NODE = "#b8b8b8"
ACTIVE_EDGE = "#4c78a8"
CONTRACTED_EDGE = "#f58518"


def _logaddexp(a: float, b: float) -> float:
    if np.isneginf(a):
        return b
    if np.isneginf(b):
        return a
    m = max(a, b)
    return m + float(np.log(np.exp(a - m) + np.exp(b - m)))


def _pack_columns(M: np.ndarray, offset: int = 0) -> list[int]:
    rows, cols = M.shape
    packed = []
    for c in range(cols):
        bits = 0
        for r in range(rows):
            if M[r, c]:
                bits |= 1 << (offset + r)
        packed.append(bits)
    return packed


def _pack_bits(bits: np.ndarray) -> int:
    out = 0
    for i, bit in enumerate(np.asarray(bits, dtype=np.uint8)):
        if bit:
            out |= 1 << i
    return out


def _trim_log_distribution(dist: dict[int, float], max_states: int | None) -> dict[int, float]:
    if max_states is None or len(dist) <= max_states:
        return dist
    keep = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:max_states]
    return dict(keep)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Torch is required for ML contraction animation. Install PyTorch in "
            "the environment running this script, or use an environment where "
            "flashbp's Torch dependencies are available."
        ) from exc
    return torch


def _resolve_torch_device(torch, device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _torch_logsumexp_by_state(torch, states, log_probs):
    """Group duplicate integer states and log-sum-exp their probabilities."""
    unique_states, inverse = torch.unique(states, sorted=True, return_inverse=True)
    n = unique_states.numel()

    # logsumexp per group = max + log(sum(exp(x - max))).  scatter_reduce keeps
    # this on the selected device, including CUDA.
    group_max = torch.full(
        (n,),
        -torch.inf,
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    if hasattr(group_max, "scatter_reduce_"):
        group_max.scatter_reduce_(0, inverse, log_probs, reduce="amax", include_self=True)
    else:
        # Older PyTorch fallback.  This is slower but preserves correctness.
        for i in range(log_probs.numel()):
            j = int(inverse[i].item())
            group_max[j] = torch.maximum(group_max[j], log_probs[i])

    shifted = torch.exp(log_probs - group_max[inverse])
    group_sum = torch.zeros(n, dtype=log_probs.dtype, device=log_probs.device)
    group_sum.scatter_add_(0, inverse, shifted)
    return unique_states, group_max + torch.log(group_sum)


def _torch_topk_states(torch, states, log_probs, max_states: int | None):
    if max_states is None or states.numel() <= max_states:
        return states, log_probs
    vals, idx = torch.topk(log_probs, k=max_states)
    return states[idx], vals


def _torch_distribution_to_dict(states, log_probs) -> dict[int, float]:
    states_cpu = states.detach().to("cpu").tolist()
    logp_cpu = log_probs.detach().to("cpu").tolist()
    return {int(s): float(lp) for s, lp in zip(states_cpu, logp_cpu)}


def _sigmoid_from_llr(llr: float) -> float:
    """Return P(error=1) for the convention LLR=log(P0/P1)."""
    if llr >= 50.0:
        return 0.0
    if llr <= -50.0:
        return 1.0
    return 1.0 / (1.0 + np.exp(llr))


def _axis_marginal_error_probs(weight: np.ndarray, parity: np.ndarray, k_axes: int) -> np.ndarray:
    """Min-sum marginal likelihood estimate per axis from a full tensor."""
    probs = np.zeros(k_axes, dtype=np.float64)
    valid = parity == 0
    for k in range(k_axes):
        bits = ((np.arange(weight.size, dtype=np.uint64) >> k) & 1).astype(bool)
        w0 = np.min(weight[valid & ~bits]) if np.any(valid & ~bits) else np.inf
        w1 = np.min(weight[valid & bits]) if np.any(valid & bits) else np.inf
        if np.isinf(w0) and np.isinf(w1):
            probs[k] = 0.5
        elif np.isinf(w0):
            probs[k] = 1.0
        elif np.isinf(w1):
            probs[k] = 0.0
        else:
            probs[k] = _sigmoid_from_llr(w1 - w0)
    return probs


def _contract_tensor_prefix(
    weight: np.ndarray,
    parity: np.ndarray,
    k_axes: int,
    contracted_axes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Min-contract the first `contracted_axes` axes.

    Returned arrays are indexed by the remaining-axis bit pattern.  The validity
    array is true when at least one contracted assignment leaves parity satisfied.
    """
    remaining_axes = k_axes - contracted_axes
    n_remaining = 1 << remaining_axes
    contracted_weight = np.full(n_remaining, np.inf, dtype=np.float64)
    contracted_valid = np.zeros(n_remaining, dtype=bool)

    for idx, w in enumerate(weight):
        rem = idx >> contracted_axes
        if parity[idx] == 0:
            contracted_valid[rem] = True
            if w < contracted_weight[rem]:
                contracted_weight[rem] = w

    return contracted_weight, contracted_valid


def _format_bits(value: int, width: int) -> str:
    if width == 0:
        return "{}"
    return format(value, f"0{width}b")


def _render_factor_graph(
    ax,
    H: np.ndarray,
    layout: dict,
    syndrome: np.ndarray,
    tensor_record: dict,
    axis_probs: np.ndarray,
    contracted_count: int,
) -> None:
    num_checks, num_vars = H.shape
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    edges = edges_from_H(H)
    nbhd_data = np.asarray(tensor_record["nbhd_data"], dtype=int)
    check_idx = int(tensor_record["check_idx"])

    active_vars = set(int(v) for v in nbhd_data)
    contracted_vars = set(int(v) for v in nbhd_data[:contracted_count])
    active_checks = {check_idx}
    for d, v in edges:
        if v in active_vars:
            active_checks.add(d)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Full Tanner connectivity, drawn as a quiet underlay before contraction.
    for d, v in edges:
        x1, y1 = var_pos[v]
        x2, y2 = check_pos[d]
        ax.plot(
            [x1, x2], [y1, y2],
            color="#e3e3e3",
            linewidth=0.35,
            alpha=0.7,
            zorder=0,
        )

    for d, v in edges:
        x1, y1 = var_pos[v]
        x2, y2 = check_pos[d]
        is_active = v in active_vars and d in active_checks
        is_contracted = v in contracted_vars
        if not (is_active or is_contracted):
            continue
        ax.plot(
            [x1, x2], [y1, y2],
            color=CONTRACTED_EDGE if is_contracted else ACTIVE_EDGE,
            linewidth=2.4 if is_contracted else 1.4,
            alpha=0.9,
            zorder=1,
        )

    base_size = layout.get("node_size", max(60.0, 4000.0 / max(num_vars, num_checks)))

    cmap = plt.get_cmap("Reds")
    var_prob_by_idx = {int(v): float(axis_probs[i]) for i, v in enumerate(nbhd_data)}
    var_faces = []
    var_edges = []
    var_lws = []
    for v in range(num_vars):
        if v in active_vars:
            var_faces.append(cmap(0.18 + 0.75 * var_prob_by_idx[v]))
            var_edges.append(CONTRACTED_EDGE if v in contracted_vars else "black")
            var_lws.append(2.8 if v in contracted_vars else 1.1)
        else:
            var_faces.append("white")
            var_edges.append(FAINT_NODE)
            var_lws.append(0.4)

    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=var_faces,
        edgecolors=var_edges,
        linewidths=var_lws,
        zorder=3,
    )

    check_faces = []
    check_edges = []
    check_lws = []
    for d in range(num_checks):
        if d in active_checks:
            check_faces.append("black" if syndrome[d] else "white")
            check_edges.append(CONTRACTED_EDGE if d == check_idx else "black")
            check_lws.append(2.8 if d == check_idx else 1.1)
        else:
            check_faces.append("white")
            check_edges.append(FAINT_NODE)
            check_lws.append(0.4)

    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=check_faces,
        edgecolors=check_edges,
        linewidths=check_lws,
        marker="s",
        zorder=3,
    )

    label_fontsize = max(4.0, min(8.0, 54.0 / np.sqrt(max(1, num_vars))))
    for axis, v in enumerate(nbhd_data):
        x, y = var_pos[int(v)]
        ax.annotate(
            f"{axis}",
            xy=(x, y),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color="black",
            zorder=4,
        )


def _render_configuration_chart(
    ax,
    tensor_record: dict,
    contracted_count: int,
    max_bars: int,
) -> None:
    weight = np.asarray(tensor_record["weight"], dtype=np.float64)
    parity = np.asarray(tensor_record["parity"], dtype=np.uint8)
    k_axes = int(round(np.log2(max(1, weight.size))))
    contracted_weight, valid = _contract_tensor_prefix(weight, parity, k_axes, contracted_count)
    remaining_axes = k_axes - contracted_count

    finite = np.isfinite(contracted_weight)
    display_weight = contracted_weight.copy()
    if np.any(finite):
        display_weight[finite] -= np.min(display_weight[finite])
    else:
        display_weight[:] = 0.0

    n = len(display_weight)
    if n > max_bars:
        finite_order = np.argsort(np.where(finite, display_weight, np.inf))
        keep = np.sort(finite_order[:max_bars])
    else:
        keep = np.arange(n)

    x = np.arange(len(keep))
    y = np.where(np.isfinite(display_weight[keep]), display_weight[keep], 0.0)
    colors = ["#4c78a8" if valid[i] else "#d9d9d9" for i in keep]
    edges = ["black" if valid[i] else "#a0a0a0" for i in keep]

    ax.bar(x, y, color=colors, edgecolor=edges, linewidth=0.6)
    ax.set_ylabel("relative cost")
    ax.set_xlabel("remaining error-vector bits")
    ax.set_title(
        f"contracted axes: {contracted_count}/{k_axes}    configs shown: {len(keep)}/{n}",
        fontsize=10,
    )
    if len(keep) <= 32:
        ax.set_xticks(x)
        ax.set_xticklabels([_format_bits(int(i), remaining_axes) for i in keep],
                           rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25)

    if n > max_bars:
        ax.text(
            0.99, 0.98,
            f"showing {max_bars} lowest-cost branches",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555",
        )


def render_tensor_contraction_frame(
    tensor_record: dict,
    iteration_record: dict,
    H: np.ndarray,
    layout: dict,
    output_path: str | Path,
    contracted_count: int,
    figsize: tuple[float, float] | None = None,
    max_bars: int = 64,
) -> None:
    """
    Render one contraction step for a recorded check tensor.

    `contracted_count` is the number of leading tensor axes to show as already
    contracted.  Frame 0 has none contracted; frame K has all axes contracted.
    """
    H = np.asarray(H)
    syndrome = np.asarray(iteration_record["syndrome"], dtype=np.uint8)
    weight = np.asarray(tensor_record["weight"], dtype=np.float64)
    parity = np.asarray(tensor_record["parity"], dtype=np.uint8)
    k_axes = int(round(np.log2(max(1, weight.size))))
    contracted_count = max(0, min(contracted_count, k_axes))
    axis_probs = _axis_marginal_error_probs(weight, parity, k_axes)

    if figsize is None:
        base = layout.get("figsize", (9.0, 7.0))
        figsize = (max(11.0, float(base[0]) * 1.45), max(6.0, float(base[1])))

    fig, (ax_graph, ax_chart) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )

    _render_factor_graph(
        ax_graph, H, layout, syndrome, tensor_record, axis_probs, contracted_count
    )
    _render_configuration_chart(ax_chart, tensor_record, contracted_count, max_bars)

    fig.suptitle(
        f"iter={int(iteration_record['iteration'])}    "
        f"check={int(tensor_record['check_idx'])}    "
        f"axes={k_axes}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def animate_tensor_contraction(
    bp,
    recording: list,
    output_dir: str | Path,
    shot_index: int = 0,
    iteration_index: int = 0,
    tensor_index: int = 0,
    framerate: float = 2.0,
    video_name: str = "tensor_contraction.mp4",
    layout: dict | None = None,
    max_bars: int = 64,
) -> Path:
    """
    Render a one-axis-per-frame contraction video for a recorded tensor.

    Requires a recording created with ``DecoderConfig(log=True,
    log_type="tensor")`` and a decoder that records tensors.
    """
    if not recording:
        raise ValueError("recording is empty; run with log=True, log_type='tensor'.")

    shot = recording[shot_index]
    iterations = shot["iterations"]
    if not iterations:
        raise ValueError(f"shot {shot_index} has no recorded iterations.")

    iteration_record = iterations[iteration_index]
    tensors = iteration_record.get("tensors", [])
    if not tensors:
        raise ValueError(
            f"iteration {iteration_index} has no tensors; use TensorLogger with TensorDecoder."
        )

    tensor_record = tensors[tensor_index]
    weight = np.asarray(tensor_record["weight"])
    k_axes = int(round(np.log2(max(1, weight.size))))

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    H = np.asarray(bp.H)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)

    for contracted_count in range(k_axes + 1):
        render_tensor_contraction_frame(
            tensor_record,
            iteration_record,
            H,
            layout,
            frames_dir / f"frame_{contracted_count:04d}.png",
            contracted_count=contracted_count,
            max_bars=max_bars,
        )

    return make_video(frames_dir, output_dir / video_name, framerate=framerate)


def _ml_prefix_distributions(
    H: np.ndarray,
    L: np.ndarray,
    error_probs: np.ndarray,
    order: list[int],
    max_states: int | None,
) -> list[dict[int, float]]:
    """Contract independent error axes into log P(accumulated syndrome, logical)."""
    num_detectors, num_errors = H.shape
    num_observables = L.shape[0]
    h_cols = _pack_columns(H, offset=0)
    l_cols = _pack_columns(L, offset=num_detectors)
    flips = [h_cols[e] | l_cols[e] for e in range(num_errors)]

    dist: dict[int, float] = {0: 0.0}
    snapshots = [dist]
    for e in order:
        p = float(np.clip(error_probs[e], 1e-300, 1.0 - 1e-300))
        log0 = float(np.log1p(-p))
        log1 = float(np.log(p))
        flip = flips[e]
        nxt: dict[int, float] = {}
        for state, logp in dist.items():
            same = logp + log0
            toggled = logp + log1
            nxt[state] = _logaddexp(nxt.get(state, -np.inf), same)
            state1 = state ^ flip
            nxt[state1] = _logaddexp(nxt.get(state1, -np.inf), toggled)
        dist = _trim_log_distribution(nxt, max_states)
        snapshots.append(dist)
    return snapshots


def _ml_prefix_distributions_torch(
    H: np.ndarray,
    L: np.ndarray,
    error_probs: np.ndarray,
    order: list[int],
    max_states: int | None,
    device: str = "auto",
    dtype: str = "float64",
) -> tuple[list[dict[int, float]], str]:
    """
    Torch-backed sparse contraction of P(accumulated syndrome, logical).

    This is exact when `max_states` is None.  When `max_states` is provided it
    becomes a beam contraction, keeping the highest-probability accumulated
    states after each axis.
    """
    torch = _import_torch()
    dev = _resolve_torch_device(torch, device)
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64

    num_detectors, num_errors = H.shape
    h_cols = _pack_columns(H, offset=0)
    l_cols = _pack_columns(L, offset=num_detectors)
    flips_np = np.array(
        [h_cols[e] | l_cols[e] for e in range(num_errors)],
        dtype=np.int64,
    )
    flips = torch.as_tensor(flips_np, dtype=torch.int64, device=dev)
    probs = torch.as_tensor(
        np.clip(error_probs, 1e-300, 1.0 - 1e-300),
        dtype=torch_dtype,
        device=dev,
    )

    states = torch.zeros(1, dtype=torch.int64, device=dev)
    log_probs = torch.zeros(1, dtype=torch_dtype, device=dev)
    snapshots = [_torch_distribution_to_dict(states, log_probs)]

    for e in order:
        p = probs[e]
        log0 = torch.log1p(-p)
        log1 = torch.log(p)
        candidate_states = torch.cat([states, torch.bitwise_xor(states, flips[e])])
        candidate_logp = torch.cat([log_probs + log0, log_probs + log1])
        states, log_probs = _torch_logsumexp_by_state(
            torch, candidate_states, candidate_logp
        )
        states, log_probs = _torch_topk_states(torch, states, log_probs, max_states)
        snapshots.append(_torch_distribution_to_dict(states, log_probs))

    return snapshots, str(dev)


def _posterior_error_probs_for_syndrome(
    H: np.ndarray,
    error_probs: np.ndarray,
    syndrome: np.ndarray,
    max_bits: int = 22,
) -> np.ndarray:
    """
    Exact posterior P(e_i=1 | syndrome) by enumeration for small DEMs.

    For larger DEMs, fall back to priors; this keeps large-code animation usable
    as a structural contraction view instead of pretending to be exact.
    """
    num_detectors, num_errors = H.shape
    priors = np.asarray(error_probs, dtype=np.float64)
    if num_errors > max_bits:
        return priors.copy()

    h_cols = _pack_columns(H)
    target = _pack_bits(syndrome)
    log_p = np.log(np.clip(priors, 1e-300, 1.0 - 1e-300))
    log_1mp = np.log1p(-np.clip(priors, 1e-300, 1.0 - 1e-300))

    total = -np.inf
    one = np.full(num_errors, -np.inf, dtype=np.float64)
    for idx in range(1 << num_errors):
        syn = 0
        lp = 0.0
        for e in range(num_errors):
            if (idx >> e) & 1:
                syn ^= h_cols[e]
                lp += log_p[e]
            else:
                lp += log_1mp[e]
        if syn != target:
            continue
        total = _logaddexp(total, lp)
        for e in range(num_errors):
            if (idx >> e) & 1:
                one[e] = _logaddexp(float(one[e]), lp)

    if np.isneginf(total):
        return priors.copy()
    return np.exp(one - total)


def _state_label(state: int, num_detectors: int, num_observables: int) -> str:
    syndrome_mask = (1 << num_detectors) - 1
    syn = state & syndrome_mask
    logical = state >> num_detectors
    return f"s={syn:0{num_detectors}b} L={logical:0{num_observables}b}"


def _bits_to_hex(bits: np.ndarray | None) -> str | None:
    if bits is None:
        return None
    value = 0
    arr = np.asarray(bits, dtype=np.uint8)
    for i, bit in enumerate(arr):
        if bit:
            value |= 1 << i
    width = max(1, (len(arr) + 3) // 4)
    text = f"{value:0{width}x}"
    return " ".join(text[i:i + 8] for i in range(0, len(text), 8))


def _bits_to_binary(bits: np.ndarray | None) -> str | None:
    if bits is None:
        return None
    return "".join(str(int(b) & 1) for b in np.asarray(bits, dtype=np.uint8))


def _render_ml_factor_graph(
    ax,
    H: np.ndarray,
    layout: dict,
    syndrome: np.ndarray,
    posterior_probs: np.ndarray,
    order: list[int],
    contracted_count: int,
    true_errors: np.ndarray | None = None,
    ml_syndrome: np.ndarray | None = None,
) -> None:
    num_checks, num_vars = H.shape
    var_pos = layout["var_pos"]
    check_pos = layout["check_pos"]
    edges = edges_from_H(H)
    contracted = set(order[:contracted_count])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Full Tanner connectivity, drawn first so uncontracted structure remains
    # visible as a thin light-gray guide.
    for d, v in edges:
        ax.plot(
            [var_pos[v][0], check_pos[d][0]],
            [var_pos[v][1], check_pos[d][1]],
            color="#e3e3e3",
            linewidth=0.35,
            alpha=0.7,
            zorder=0,
        )

    for d, v in edges:
        is_contracted = v in contracted
        if not is_contracted:
            continue
        ax.plot(
            [var_pos[v][0], check_pos[d][0]],
            [var_pos[v][1], check_pos[d][1]],
            color=CONTRACTED_EDGE,
            linewidth=1.8,
            alpha=0.85,
            zorder=1,
        )

    base_size = layout.get("node_size", max(60.0, 4000.0 / max(num_vars, num_checks)))
    cmap = plt.get_cmap("Reds")
    var_faces = [cmap(0.15 + 0.8 * float(posterior_probs[v])) for v in range(num_vars)]
    var_edges = []
    var_lws = []
    for v in range(num_vars):
        if true_errors is not None and true_errors[v]:
            var_edges.append("#1f77b4")
            var_lws.append(3.0)
        elif v in contracted:
            var_edges.append(CONTRACTED_EDGE)
            var_lws.append(2.8)
        else:
            var_edges.append("black")
            var_lws.append(1.0)
    ax.scatter(
        [var_pos[v][0] for v in range(num_vars)],
        [var_pos[v][1] for v in range(num_vars)],
        s=base_size,
        c=var_faces,
        edgecolors=var_edges,
        linewidths=var_lws,
        zorder=3,
    )

    check_faces = []
    check_edges = []
    for d in range(num_checks):
        if ml_syndrome is not None and ml_syndrome[d]:
            check_faces.append("#1f77b4")
            check_edges.append("#1f77b4")
        elif syndrome[d]:
            check_faces.append("black")
            check_edges.append("black")
        else:
            check_faces.append("white")
            check_edges.append("black")
    ax.scatter(
        [check_pos[d][0] for d in range(num_checks)],
        [check_pos[d][1] for d in range(num_checks)],
        s=base_size,
        c=check_faces,
        edgecolors=check_edges,
        linewidths=1.0,
        marker="s",
        zorder=3,
    )

    label_fontsize = max(4.0, min(8.0, 54.0 / np.sqrt(max(1, num_vars))))
    for v in range(num_vars):
        x, y = var_pos[v]
        ax.annotate(
            f"{v}",
            xy=(x, y),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color="black",
            zorder=4,
        )


def _render_ml_distribution_chart(
    ax,
    dist: dict[int, float],
    num_detectors: int,
    num_observables: int,
    syndrome: np.ndarray,
    max_bars: int,
) -> None:
    target = _pack_bits(syndrome)
    syndrome_mask = (1 << num_detectors) - 1
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:max_bars]
    if not items:
        ax.text(0.5, 0.5, "empty distribution", ha="center", va="center")
        ax.axis("off")
        return

    logs = np.array([lp for _, lp in items], dtype=np.float64)
    heights = np.exp(logs - float(np.max(logs)))
    colors = [
        "#4c78a8" if (state & syndrome_mask) == target else "#d9d9d9"
        for state, _ in items
    ]
    x = np.arange(len(items))
    ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("relative probability")
    ax.set_xlabel("accumulated state")
    ax.set_title(f"top {len(items)} accumulated syndrome/logical states", fontsize=10)

    if len(items) <= 24 and num_detectors <= 10:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_state_label(state, num_detectors, num_observables) for state, _ in items],
            rotation=90,
            fontsize=7,
        )
    else:
        ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25)


def _render_ml_tensor_heatmap(
    ax,
    dist: dict[int, float],
    num_detectors: int | None = None,
    logical_bit: int | None = None,
    title: str = "contracted tensor heatmap",
    max_axis: int = 128,
) -> None:
    if not dist:
        ax.text(0.5, 0.5, "empty tensor", ha="center", va="center")
        ax.axis("off")
        return

    states = np.array(list(dist.keys()), dtype=np.uint64)
    logs = np.array(list(dist.values()), dtype=np.float64)
    finite = np.isfinite(logs)
    states = states[finite]
    logs = logs[finite]
    if states.size == 0:
        ax.text(0.5, 0.5, "no finite mass", ha="center", va="center", fontsize=8)
        ax.axis("off")
        return
    if logical_bit is not None and num_detectors is not None:
        syndrome_mask = np.uint64((1 << num_detectors) - 1)
        values = (states >> np.uint64(num_detectors + logical_bit)) & np.uint64(1)
        states = (states & syndrome_mask) | (values << np.uint64(num_detectors))

    collapsed_logs: dict[int, float] = {}
    for state, logp in zip(states, logs):
        key = int(state)
        collapsed_logs[key] = _logaddexp(collapsed_logs.get(key, -np.inf), float(logp))

    keys = np.array(list(collapsed_logs.keys()), dtype=np.uint64)
    key_logs = np.array(list(collapsed_logs.values()), dtype=np.float64)
    probs = _normalized_log_probs(key_logs)

    max_key = int(keys.max()) if keys.size else 0
    active_bits = [
        bit for bit in range(max(1, max_key.bit_length()))
        if np.any(((keys >> np.uint64(bit)) & np.uint64(1)) != 0)
    ]
    if not active_bits:
        active_bits = [0]

    axis_scores = []
    for bit in active_bits:
        mask = ((keys >> np.uint64(bit)) & np.uint64(1)).astype(bool)
        p0 = float(np.sum(probs[~mask]))
        p1 = float(np.sum(probs[mask]))
        axis_scores.append((p0 - p1, p0, -bit, bit))
    axis_order = [bit for _, _, _, bit in sorted(axis_scores, reverse=True)]

    max_bits = max(1, int(np.floor(np.log2(max_axis * max_axis))))
    truncated = len(axis_order) > max_bits
    displayed_order = axis_order[:max_bits] if truncated else axis_order
    display_bits = len(displayed_order)
    col_bits = (display_bits + 1) // 2
    row_bits = display_bits - col_bits
    cols = 1 << max(0, col_bits)
    rows = 1 << max(0, row_bits)

    positive = probs[probs > 0.0]
    floor = 1e-5
    if positive.size:
        floor = max(float(positive.max()) * 1e-5, 1e-12)
    grid = np.full((rows, cols), floor, dtype=np.float64)
    for key, prob in zip(keys, probs):
        idx = 0
        for rank, in_bit in enumerate(displayed_order):
            out_bit = display_bits - 1 - rank
            if (int(key) >> in_bit) & 1:
                idx |= 1 << out_bit
        col = idx & (cols - 1)
        row = idx >> col_bits
        grid[row, col] += max(float(prob), floor)
    grid = np.maximum(grid - floor, floor)

    cmap = plt.get_cmap("viridis")
    norm = None
    if positive.size:
        vmax = float(np.percentile(positive, 98))
        if vmax <= floor:
            vmax = float(positive.max())
        if vmax > floor:
            norm = LogNorm(vmin=floor, vmax=vmax)
    ax.imshow(
        grid,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(
        title + (" (axis-sorted/top)" if truncated else " (axis-sorted)"),
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("lower sorted axes", fontsize=8)
    ax.set_ylabel("higher sorted axes", fontsize=8)


def _normalized_log_probs(logs: np.ndarray) -> np.ndarray:
    logs = np.asarray(logs, dtype=np.float64)
    if np.all(np.isneginf(logs)):
        return np.zeros_like(logs, dtype=np.float64)
    shifted = np.exp(logs - float(np.max(logs)))
    shifted[np.isneginf(logs)] = 0.0
    total = float(np.sum(shifted))
    if total <= 0.0:
        return np.zeros_like(logs, dtype=np.float64)
    return shifted / total


def _logical_bit_marginal_logs(
    class_logs: np.ndarray,
    bit: int,
) -> np.ndarray:
    out = np.full(2, -np.inf, dtype=np.float64)
    for cls, logp in enumerate(class_logs):
        value = (cls >> bit) & 1
        out[value] = _logaddexp(float(out[value]), float(logp))
    return out


def _class_marginal_logs_from_dist(
    dist: dict[int, float],
    num_detectors: int,
    num_observables: int,
) -> np.ndarray:
    n_classes = 1 << num_observables
    out = np.full(n_classes, -np.inf, dtype=np.float64)
    for state, logp in dist.items():
        if not np.isfinite(logp):
            continue
        logical = int(state) >> num_detectors
        logical &= n_classes - 1
        out[logical] = _logaddexp(float(out[logical]), float(logp))
    return out


def _render_binary_logical_panel(
    ax,
    logs: np.ndarray,
    title: str,
    ylabel: bool,
) -> None:
    heights = _normalized_log_probs(logs)
    best = None if np.all(np.isneginf(logs)) else int(np.argmax(logs))
    colors = ["#54a24b" if best == i else "#9ecae9" for i in range(2)]
    ax.bar([0, 1], heights, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=8)
    ax.set_xlabel("logical value", fontsize=8)
    if ylabel:
        ax.set_ylabel("probability", fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.grid(axis="y", alpha=0.25)


def _ml_summary_lines(
    num_observables: int,
    best: int | None,
    true_errors: np.ndarray | None = None,
    predicted_errors: np.ndarray | None = None,
    true_logical: np.ndarray | None = None,
    predicted_logical: np.ndarray | None = None,
    true_syndrome_ok: bool | None = None,
    predicted_syndrome_ok: bool | None = None,
) -> list[str]:
    lines = []
    true_hex = _bits_to_hex(true_errors)
    pred_hex = _bits_to_hex(predicted_errors)
    if true_hex is not None:
        lines.append(f"sampled error: 0x{true_hex}")
    if pred_hex is not None:
        lines.append(f"ML representative: 0x{pred_hex}")
    true_logical_text = _bits_to_binary(true_logical)
    pred_logical_text = _bits_to_binary(predicted_logical)
    if true_logical_text is not None or pred_logical_text is not None:
        lines.append(
            f"logical sampled/rep: "
            f"{true_logical_text if true_logical_text is not None else '?'} / "
            f"{pred_logical_text if pred_logical_text is not None else '?'}"
        )
    checks = []
    if true_syndrome_ok is not None:
        checks.append(f"sampled syndrome {'ok' if true_syndrome_ok else 'bad'}")
    if predicted_syndrome_ok is not None:
        checks.append(f"rep syndrome {'ok' if predicted_syndrome_ok else 'bad'}")
    if checks:
        lines.append(", ".join(checks))
    if best is not None:
        best_label = format(best, f"0{num_observables}b")
        lines.append(f"ML class: {best_label}")
    return lines


def _render_ml_class_chart(
    ax,
    dist: dict[int, float],
    num_detectors: int,
    num_observables: int,
    syndrome: np.ndarray,
    class_log_probs: np.ndarray | None = None,
    true_errors: np.ndarray | None = None,
    predicted_errors: np.ndarray | None = None,
    true_logical: np.ndarray | None = None,
    predicted_logical: np.ndarray | None = None,
    true_syndrome_ok: bool | None = None,
    predicted_syndrome_ok: bool | None = None,
    draw_summary: bool = False,
    title: str = "final ML slice at observed syndrome",
) -> None:
    n_classes = 1 << num_observables
    if class_log_probs is None:
        target = _pack_bits(syndrome)
        syndrome_mask = (1 << num_detectors) - 1
        class_logs = np.full(n_classes, -np.inf, dtype=np.float64)
        for state, logp in dist.items():
            if (state & syndrome_mask) != target:
                continue
            logical = state >> num_detectors
            class_logs[logical] = _logaddexp(float(class_logs[logical]), logp)
    else:
        class_logs = np.asarray(class_log_probs, dtype=np.float64)
        if np.all(np.isneginf(class_logs)):
            class_logs = _class_marginal_logs_from_dist(
                dist, num_detectors, num_observables
            )
            title = title + " (unconditioned)"

    best = None if np.all(np.isneginf(class_logs)) else int(np.argmax(class_logs))

    if num_observables == 2:
        ax.set_title(title, fontsize=10, pad=12)
        ax.axis("off")
        ax_x = ax.inset_axes([0.02, 0.17, 0.45, 0.72])
        ax_z = ax.inset_axes([0.53, 0.17, 0.45, 0.72])
        _render_binary_logical_panel(
            ax_x, _logical_bit_marginal_logs(class_logs, 0), "X logical", True
        )
        _render_binary_logical_panel(
            ax_z, _logical_bit_marginal_logs(class_logs, 1), "Z logical", False
        )
    else:
        heights = _normalized_log_probs(class_logs)

        x = np.arange(n_classes)
        colors = ["#54a24b" if best is not None and i == best else "#9ecae9"
                  for i in range(n_classes)]
        ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel("probability")
        ax.set_xlabel("logical class")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.0, 1.0)
        if n_classes <= 32:
            ax.set_xticks(x)
            ax.set_xticklabels([format(i, f"0{num_observables}b") for i in x],
                               rotation=90 if n_classes > 8 else 0,
                               fontsize=7)
        else:
            ax.set_xticks([])
        ax.grid(axis="y", alpha=0.25)

    if draw_summary:
        lines = _ml_summary_lines(
            num_observables,
            best,
            true_errors=true_errors,
            predicted_errors=predicted_errors,
            true_logical=true_logical,
            predicted_logical=predicted_logical,
            true_syndrome_ok=true_syndrome_ok,
            predicted_syndrome_ok=predicted_syndrome_ok,
        )
        ax.text(
            0.02, -0.18,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            clip_on=False,
        )


def render_ml_contraction_frame(
    H: np.ndarray,
    L: np.ndarray,
    error_probs: np.ndarray,
    syndrome: np.ndarray,
    layout: dict,
    output_path: str | Path,
    contracted_count: int,
    order: list[int] | None = None,
    prefix_distributions: list[dict[int, float]] | None = None,
    posterior_probs: np.ndarray | None = None,
    figsize: tuple[float, float] | None = None,
    max_bars: int = 64,
    max_states: int | None = None,
    contraction_backend: str = "torch",
    contraction_device: str = "auto",
    contraction_dtype: str = "float64",
    resolved_device: str | None = None,
    true_errors: np.ndarray | None = None,
    predicted_errors: np.ndarray | None = None,
    class_log_probs: np.ndarray | None = None,
    step_class_log_probs: np.ndarray | None = None,
    heatmap_distributions: dict[int | None, dict[int, float]] | None = None,
) -> None:
    """Render one frame of the ML coset-sum contraction."""
    H = np.asarray(H, dtype=np.uint8)
    L = np.asarray(L, dtype=np.uint8)
    error_probs = np.asarray(error_probs, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    num_detectors, num_errors = H.shape
    num_observables = L.shape[0]
    if order is None:
        order = list(range(num_errors))
    contracted_count = max(0, min(contracted_count, len(order)))
    if prefix_distributions is None:
        if contraction_backend == "torch":
            prefix_distributions, resolved_device = _ml_prefix_distributions_torch(
                H, L, error_probs, order, max_states,
                device=contraction_device,
                dtype=contraction_dtype,
            )
        elif contraction_backend == "python":
            prefix_distributions = _ml_prefix_distributions(H, L, error_probs, order, max_states)
            resolved_device = "cpu/python"
        else:
            raise ValueError("contraction_backend must be 'torch' or 'python'.")
    if posterior_probs is None:
        posterior_probs = _posterior_error_probs_for_syndrome(H, error_probs, syndrome)

    final_frame = contracted_count == len(order)
    ml_syndrome = None
    true_logical = None
    predicted_logical = None
    true_syndrome_ok = None
    predicted_syndrome_ok = None
    if final_frame:
        if true_errors is not None:
            true_arr = np.asarray(true_errors, dtype=np.uint8)
            true_syndrome = (H.astype(np.int32) @ true_arr.astype(np.int32)) % 2
            true_logical = (L.astype(np.int32) @ true_arr.astype(np.int32)) % 2
            true_syndrome_ok = bool(np.array_equal(true_syndrome, syndrome))
        if predicted_errors is not None:
            pred_arr = np.asarray(predicted_errors, dtype=np.uint8)
            pred_syndrome = (H.astype(np.int32) @ pred_arr.astype(np.int32)) % 2
            predicted_logical = (L.astype(np.int32) @ pred_arr.astype(np.int32)) % 2
            predicted_syndrome_ok = bool(np.array_equal(pred_syndrome, syndrome))
            ml_syndrome = pred_syndrome.astype(np.uint8)

    if figsize is None:
        base = layout.get("figsize", (9.0, 7.0))
        figsize = (max(11.0, float(base[0]) * 1.45), max(6.0, float(base[1])))

    fig, (ax_graph, ax_chart) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )
    _render_ml_factor_graph(
        ax_graph, H, layout, syndrome, posterior_probs, order, contracted_count,
        true_errors=true_errors,
        ml_syndrome=ml_syndrome,
    )

    dist = prefix_distributions[contracted_count]
    chart_ax = ax_chart
    heatmap_axes = []
    summary_lines: list[str] = []
    if dist:
        ax_chart.axis("off")
        has_bottom_text = final_frame and (true_errors is not None or predicted_errors is not None)
        chart_ax = ax_chart.inset_axes(
            [0.00, 0.52, 1.00, 0.46] if has_bottom_text else [0.00, 0.40, 1.00, 0.58]
        )
        if num_observables == 2:
            heatmap_y = 0.18 if has_bottom_text else 0.03
            heatmap_h = 0.26 if has_bottom_text else 0.30
            heatmap_axes = [
                (ax_chart.inset_axes([0.04, heatmap_y, 0.43, heatmap_h]), 0, "X heatmap"),
                (ax_chart.inset_axes([0.53, heatmap_y, 0.43, heatmap_h]), 1, "Z heatmap"),
            ]
        else:
            heatmap_y = 0.18 if has_bottom_text else 0.03
            heatmap_h = 0.26 if has_bottom_text else 0.30
            heatmap_axes = [
                (ax_chart.inset_axes([0.08, heatmap_y, 0.84, heatmap_h]), None,
                 "contracted tensor heatmap"),
            ]
    if final_frame:
        if class_log_probs is None:
            best = None
        else:
            class_logs = np.asarray(class_log_probs, dtype=np.float64)
            best = None if class_logs.size == 0 or np.all(np.isneginf(class_logs)) else int(np.argmax(class_logs))
        summary_lines = _ml_summary_lines(
            num_observables,
            best,
            true_errors=true_errors,
            predicted_errors=predicted_errors,
            true_logical=true_logical,
            predicted_logical=predicted_logical,
            true_syndrome_ok=true_syndrome_ok,
            predicted_syndrome_ok=predicted_syndrome_ok,
        )
        _render_ml_class_chart(
            chart_ax, dist, num_detectors, num_observables, syndrome,
            class_log_probs=class_log_probs,
            true_errors=true_errors,
            predicted_errors=predicted_errors,
            true_logical=true_logical,
            predicted_logical=predicted_logical,
            true_syndrome_ok=true_syndrome_ok,
            predicted_syndrome_ok=predicted_syndrome_ok,
            draw_summary=False,
        )
    elif step_class_log_probs is not None:
        _render_ml_class_chart(
            chart_ax, dist, num_detectors, num_observables, syndrome,
            class_log_probs=step_class_log_probs,
            title="current observed-syndrome logical slice",
        )
    else:
        _render_ml_distribution_chart(
            chart_ax, dist, num_detectors, num_observables, syndrome, max_bars
        )
    for heatmap_ax, logical_bit, heatmap_title in heatmap_axes:
        heatmap_dist = dist
        if heatmap_distributions is not None:
            heatmap_dist = heatmap_distributions.get(logical_bit, dist)
        _render_ml_tensor_heatmap(
            heatmap_ax,
            heatmap_dist,
            num_detectors=num_detectors,
            logical_bit=logical_bit,
            title=heatmap_title,
        )
    if summary_lines:
        ax_chart.text(
            0.02,
            0.02,
            "\n".join(summary_lines),
            transform=ax_chart.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            family="monospace",
            clip_on=False,
        )

    fig.suptitle(
        f"maximum-likelihood contraction    axes={contracted_count}/{len(order)}"
        + (f"    backend={resolved_device}" if resolved_device else ""),
        fontsize=12,
    )
    if contracted_count == len(order):
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def animate_ml_contraction(
    bp,
    syndrome,
    output_dir: str | Path,
    framerate: float = 2.0,
    video_name: str = "ml_contraction.mp4",
    layout: dict | None = None,
    order: list[int] | None = None,
    max_bars: int = 64,
    max_states: int | None = None,
    posterior_max_bits: int = 22,
    contraction_backend: str = "torch",
    device: str = "auto",
    dtype: str = "float64",
    true_errors: np.ndarray | None = None,
    predicted_errors: np.ndarray | None = None,
) -> Path:
    """
    Animate the ML decoder's coset-sum contraction over DEM error axes.

    The distribution being contracted is P(accumulated syndrome, logical class)
    after summing over a prefix of the independent error variables.
    """
    H = np.asarray(bp.H, dtype=np.uint8)
    L = np.asarray(bp.L, dtype=np.uint8)
    error_probs = np.asarray(bp.error_probs, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)
    if order is None:
        order = list(range(num_vars))

    if contraction_backend == "torch":
        prefix_distributions, resolved_device = _ml_prefix_distributions_torch(
            H, L, error_probs, order, max_states, device=device, dtype=dtype
        )
    elif contraction_backend == "python":
        prefix_distributions = _ml_prefix_distributions(H, L, error_probs, order, max_states)
        resolved_device = "cpu/python"
    else:
        raise ValueError("contraction_backend must be 'torch' or 'python'.")
    posterior_probs = _posterior_error_probs_for_syndrome(
        H, error_probs, syndrome, max_bits=posterior_max_bits
    )

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for contracted_count in range(len(order) + 1):
        render_ml_contraction_frame(
            H,
            L,
            error_probs,
            syndrome,
            layout,
            frames_dir / f"frame_{contracted_count:04d}.png",
            contracted_count=contracted_count,
            order=order,
            prefix_distributions=prefix_distributions,
            posterior_probs=posterior_probs,
            max_bars=max_bars,
            max_states=max_states,
            contraction_backend=contraction_backend,
            contraction_device=device,
            contraction_dtype=dtype,
            resolved_device=resolved_device,
            true_errors=true_errors,
            predicted_errors=predicted_errors,
        )

    return make_video(frames_dir, output_dir / video_name, framerate=framerate)


def _ml_recording_prefix_distributions(shot: dict) -> list[dict[int, float]]:
    prefix_distributions: list[dict[int, float]] = []
    for step in shot["steps"]:
        states = np.asarray(step["states"], dtype=np.int64)
        log_probs = np.asarray(step["log_probs"], dtype=np.float64)
        prefix_distributions.append(
            {int(s): float(lp) for s, lp in zip(states, log_probs)}
        )
    return prefix_distributions


def _ml_recording_class_distributions(shot: dict) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    for step in shot["steps"]:
        values = np.asarray(step.get("class_log_probs", []), dtype=np.float64)
        out.append(values if values.size else None)
    return out


def _error_component_logical_bits(H: np.ndarray, L: np.ndarray) -> tuple[list[int], dict[int, list[int]]]:
    num_checks, num_errors = H.shape
    total = num_errors + num_checks
    adj: list[list[int]] = [[] for _ in range(total)]
    for d in range(num_checks):
        for e in np.flatnonzero(H[d]):
            check_node = num_errors + int(d)
            error_node = int(e)
            adj[error_node].append(check_node)
            adj[check_node].append(error_node)

    comp_of_error = [-1] * num_errors
    seen_all = set()
    comp_id = 0
    for start in range(total):
        if start in seen_all:
            continue
        if not adj[start]:
            continue
        stack = [start]
        seen = {start}
        errors: list[int] = []
        while stack:
            node = stack.pop()
            if node < num_errors:
                errors.append(node)
            for nxt in adj[node]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        seen_all.update(seen)
        for e in errors:
            comp_of_error[e] = comp_id
        comp_id += 1

    comp_bits: dict[int, list[int]] = {}
    for e, comp in enumerate(comp_of_error):
        if comp < 0:
            continue
        for bit in np.flatnonzero(L[:, e]):
            comp_bits.setdefault(comp, [])
            bit_i = int(bit)
            if bit_i not in comp_bits[comp]:
                comp_bits[comp].append(bit_i)
    return comp_of_error, comp_bits


def _ml_recording_heatmap_distributions(
    H: np.ndarray,
    L: np.ndarray,
    shot: dict,
    prefix_distributions: list[dict[int, float]],
) -> list[dict[int | None, dict[int, float]]]:
    num_observables = L.shape[0]
    if num_observables != 2:
        return [{None: dist} for dist in prefix_distributions]

    comp_of_error, comp_bits = _error_component_logical_bits(H, L)
    steps = shot["steps"]
    carried: dict[int | None, dict[int, float]] = {
        bit: prefix_distributions[0] for bit in range(num_observables)
    }
    frames: list[dict[int | None, dict[int, float]]] = []
    for i, dist in enumerate(prefix_distributions):
        if i > 0:
            error_idx = int(steps[i].get("error_idx", -1))
            if 0 <= error_idx < len(comp_of_error):
                comp = comp_of_error[error_idx]
                bits = comp_bits.get(comp, [])
                if not bits:
                    bits = list(range(num_observables))
                for bit in bits:
                    carried[int(bit)] = dist
        frames.append(dict(carried))
    return frames


def animate_ml_contraction_recording(
    bp,
    recording: list,
    output_dir: str | Path,
    shot_index: int = 0,
    framerate: float = 2.0,
    video_name: str = "ml_contraction.mp4",
    layout: dict | None = None,
    max_bars: int = 64,
    posterior_max_bits: int = 22,
    true_errors: np.ndarray | None = None,
    predicted_errors: np.ndarray | None = None,
) -> Path:
    """Animate ML contraction frames from C++ MLLogger recording data."""
    if not recording:
        raise ValueError("recording is empty; run with log=True, log_type='ml'.")

    shot = recording[shot_index]
    steps = shot.get("steps", [])
    if not steps:
        raise ValueError(
            "ML recording has no contraction steps. Rebuild/reinstall flashbp "
            "after the MLLogger changes, and construct the decoder with "
            "DecoderConfig(decoder='ml', log=True, log_type='ml')."
        )

    H = np.asarray(bp.H, dtype=np.uint8)
    L = np.asarray(bp.L, dtype=np.uint8)
    error_probs = np.asarray(bp.error_probs, dtype=np.float64)
    syndrome = np.asarray(shot["syndrome"], dtype=np.uint8)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)

    prefix_distributions = _ml_recording_prefix_distributions(shot)
    step_class_log_probs = _ml_recording_class_distributions(shot)
    heatmap_distributions = _ml_recording_heatmap_distributions(
        H, L, shot, prefix_distributions
    )
    class_log_probs = np.asarray(shot.get("class_log_probs", []), dtype=np.float64)
    if class_log_probs.size == 0:
        class_log_probs = None
    # Step 0 is the initial state; subsequent steps carry error_idx.
    order = [int(step["error_idx"]) for step in steps[1:]]
    posterior_probs = _posterior_error_probs_for_syndrome(
        H, error_probs, syndrome, max_bits=posterior_max_bits
    )

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for contracted_count in range(len(prefix_distributions)):
        render_ml_contraction_frame(
            H,
            L,
            error_probs,
            syndrome,
            layout,
            frames_dir / f"frame_{contracted_count:04d}.png",
            contracted_count=contracted_count,
            order=order,
            prefix_distributions=prefix_distributions,
            posterior_probs=posterior_probs,
            max_bars=max_bars,
            resolved_device=str(shot.get("device", "cpp")),
            true_errors=true_errors,
            predicted_errors=predicted_errors,
            class_log_probs=class_log_probs,
            step_class_log_probs=step_class_log_probs[contracted_count],
            heatmap_distributions=heatmap_distributions[contracted_count],
        )

    return make_video(frames_dir, output_dir / video_name, framerate=framerate)
