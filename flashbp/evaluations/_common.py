from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import flashbp
from flashbp import DecoderConfig
from flashbp.codes import BBCode, SteaneCode, SurfaceCode
from flashbp.animation import bb_torus_layout, surface_code_layout


CODES = {
    "steane": SteaneCode,
    "smbb": BBCode.smbb_code,
    "hbb": BBCode.hbb_code,
    "gross": BBCode.gross_code,
    "bb_144": BBCode.bb_144_12_12,
    "surface_3": lambda: SurfaceCode(3),
    "surface_5": lambda: SurfaceCode(5),
    "surface_7": lambda: SurfaceCode(7),
    "surface_9": lambda: SurfaceCode(9),
}


GBP_POLICY_KEYS = {
    "gbp": "check_neighborhood",
    "gbp-check": "check_neighborhood",
    "gbp-manual": "manual_groups",
    "gbp-groups": "manual_groups",
    "gbp-cycles": "short_cycles",
    "gbp-cycles-any": "short_cycles_any_active",
    "gbp-cycles-all": "short_cycles_all_active",
    "gbp-union-cycles": "short_cycles_union",
    "gbp-union-cycles-any": "short_cycles_union_any_active",
    "gbp-union-cycles-all": "short_cycles_union_all_active",
}

GBP_BACKEND_KEYS = {
    "cpu": "dense_cpu",
    "dense": "dense_cpu",
    "dense-cpu": "dense_cpu",
    "dense_cpu": "dense_cpu",
    "sparse": "sparse_cpu",
    "sparse-cpu": "sparse_cpu",
    "sparse_cpu": "sparse_cpu",
    "cuda": "torch_cuda",
    "torch": "torch_cuda",
    "torch-cuda": "torch_cuda",
    "torch_cuda": "torch_cuda",
}

BP_OSD_KEYS = {"bp-osd", "bposd", "bp_osd"}


def parse_int_budget(text: str) -> int:
    clean = text.replace("_", "").strip().lower()
    if clean.startswith("2^"):
        return 1 << int(clean[2:])
    suffixes = {"k": 10, "m": 20, "g": 30}
    if clean[-1:] in suffixes:
        return int(float(clean[:-1]) * (1 << suffixes[clean[-1]]))
    return int(clean, 0)


def apply_gbp_extra_field(cfg: DecoderConfig, field: str, spec: str) -> None:
    if not field:
        return
    if field in GBP_BACKEND_KEYS:
        cfg.gbp_backend = GBP_BACKEND_KEYS[field]
        return
    if field.startswith(("states=", "max_states=", "max-states=")):
        _, value = field.split("=", 1)
        cfg.gbp_max_states = parse_int_budget(value)
        return
    if field.startswith(("boost=", "osc_boost=", "oscillation_boost=")):
        _, value = field.split("=", 1)
        cfg.gbp_oscillation_boost = float(value)
        return
    if field.startswith(("boost_cap=", "boost-cap=", "oscillation_boost_cap=")):
        _, value = field.split("=", 1)
        cfg.gbp_oscillation_boost_cap = float(value)
        return
    if field.startswith(("groups=", "manual_groups=", "manual-groups=")):
        _, value = field.split("=", 1)
        path = Path(value)
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        cfg.gbp_manual_groups = (
            loaded.get("groups", loaded) if isinstance(loaded, dict) else loaded
        )
        return
    if field in ("no_single_checks", "no-single-checks", "no_checks", "no-checks"):
        cfg.gbp_manual_add_single_checks = False
        return
    if field in ("single_checks", "single-checks"):
        cfg.gbp_manual_add_single_checks = True
        return
    raise ValueError(f"unknown GBP field {field!r} in decoder spec {spec!r}")


@dataclass
class DecoderSpec:
    label: str
    backend: str
    config: DecoderConfig | None = None
    osd_order: int = 0


def parse_decoder_spec(spec: str) -> DecoderSpec:
    parts = spec.split(":")
    name = parts[0]

    if name in BP_OSD_KEYS:
        osd_order = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if len(parts) > 2:
            raise ValueError(f"too many ':' fields in decoder spec {spec!r}")
        return DecoderSpec(
            label=f"bp-osd:{osd_order}",
            backend="ldpc_bp_osd",
            osd_order=osd_order,
        )

    if name in GBP_POLICY_KEYS:
        cfg = DecoderConfig(decoder="gbp")
        cfg.region_policy = GBP_POLICY_KEYS[name]
        if len(parts) > 1 and parts[1]:
            if parts[1].lstrip("+-").isdigit():
                cfg.degree = int(parts[1])
            elif parts[1] in GBP_BACKEND_KEYS or "=" in parts[1]:
                apply_gbp_extra_field(cfg, parts[1], spec)
            else:
                cfg.region_policy = parts[1]
        if len(parts) > 2 and parts[2]:
            if parts[2] in GBP_BACKEND_KEYS:
                cfg.gbp_backend = GBP_BACKEND_KEYS[parts[2]]
            elif "=" in parts[2]:
                apply_gbp_extra_field(cfg, parts[2], spec)
            else:
                cfg.region_policy = parts[2]
        if len(parts) > 3 and parts[3]:
            apply_gbp_extra_field(cfg, parts[3], spec)
        if len(parts) > 4 and parts[4]:
            apply_gbp_extra_field(cfg, parts[4], spec)
        if len(parts) > 5 and parts[5]:
            apply_gbp_extra_field(cfg, parts[5], spec)
        if len(parts) > 6:
            raise ValueError(f"too many ':' fields in decoder spec {spec!r}")
        backend_suffix = (
            "" if cfg.gbp_backend == "dense_cpu" else f":{cfg.gbp_backend}"
        )
        states_suffix = (
            ""
            if cfg.gbp_max_states == (1 << 22)
            else f":states={cfg.gbp_max_states}"
        )
        boost_suffix = (
            ""
            if cfg.gbp_oscillation_boost == 1.0
            else f":boost={cfg.gbp_oscillation_boost:g}"
        )
        cap_suffix = (
            ""
            if cfg.gbp_oscillation_boost_cap == 64.0
            else f":boost_cap={cfg.gbp_oscillation_boost_cap:g}"
        )
        return DecoderSpec(
            label=(
                f"gbp-{cfg.region_policy}:{cfg.degree}{backend_suffix}"
                f"{states_suffix}{boost_suffix}{cap_suffix}"
            ),
            backend="flashbp",
            config=cfg,
        )

    cfg = DecoderConfig(decoder=name)
    if len(parts) > 1 and parts[1]:
        cfg.degree = int(parts[1])
    if len(parts) > 2:
        raise ValueError(f"too many ':' fields in decoder spec {spec!r}")
    return DecoderSpec(label=spec, backend="flashbp", config=cfg)


def dem_matrices(dem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_detectors = int(dem.num_detectors)
    num_observables = int(dem.num_observables)
    flat = dem.flattened()
    num_errors = sum(1 for instr in flat if instr.type == "error")

    H = np.zeros((num_detectors, num_errors), dtype=np.uint8)
    L = np.zeros((num_observables, num_errors), dtype=np.uint8)
    error_probs = np.zeros(num_errors, dtype=np.float64)

    e = 0
    for instr in flat:
        if instr.type != "error":
            continue
        error_probs[e] = float(instr.args_copy()[0])
        for target in instr.targets_copy():
            if target.is_separator():
                continue
            val = int(target.val)
            if target.is_relative_detector_id():
                H[val, e] = 1
            elif target.is_logical_observable_id():
                L[val, e] = 1
        e += 1
    return H, L, error_probs


def load_bp_osd_decoder_class():
    try:
        from ldpc import BpOsdDecoder
        return BpOsdDecoder
    except ImportError:
        pass

    try:
        from ldpc.bposd_decoder import BpOsdDecoder
        return BpOsdDecoder
    except ImportError as exc:
        raise ImportError(
            "decoder 'bp-osd' requires Roffe's optional ldpc package. "
            "Install it with `python -m pip install -U ldpc pymatching`."
        ) from exc


class LdpcBpOsdRunner:
    def __init__(self, dem, max_iter: int, osd_order: int = 0):
        BpOsdDecoder = load_bp_osd_decoder_class()
        self.H, self.L, self.error_probs = dem_matrices(dem)
        self.num_detectors, self.num_errors = self.H.shape
        self.num_observables = self.L.shape[0]
        self.decoder = self._construct_decoder(BpOsdDecoder, max_iter, osd_order)

    def _construct_decoder(self, cls, max_iter: int, osd_order: int):
        error_channel = self.error_probs.astype(float).tolist()
        osd_method = "OSD_0" if osd_order == 0 else "OSD_CS"
        attempts = (
            ((), {
                "pcm": self.H,
                "error_channel": error_channel,
                "max_iter": max_iter,
                "bp_method": "minimum_sum",
                "osd_method": osd_method,
                "osd_order": osd_order,
                "input_vector_type": "syndrome",
            }),
            ((self.H,), {
                "error_channel": error_channel,
                "max_iter": max_iter,
                "bp_method": "minimum_sum",
                "osd_method": osd_method,
                "osd_order": osd_order,
                "input_vector_type": "syndrome",
            }),
            ((), {
                "pcm": self.H,
                "error_rate": float(np.mean(self.error_probs)),
                "max_iter": max_iter,
                "bp_method": "minimum_sum",
                "osd_method": osd_method,
                "osd_order": osd_order,
                "input_vector_type": "syndrome",
            }),
            ((), {
                "parity_check_matrix": self.H,
                "channel_probs": error_channel,
                "max_iter": max_iter,
                "bp_method": "ms",
                "osd_method": "osd_cs",
                "osd_order": osd_order,
            }),
            ((self.H,), {
                "error_rate": float(np.mean(self.error_probs)),
                "max_iter": max_iter,
                "bp_method": "ms",
                "osd_method": "osd_cs",
                "osd_order": osd_order,
            }),
        )
        last_error = None
        for args, kwargs in attempts:
            try:
                return cls(*args, **kwargs)
            except TypeError as exc:
                last_error = exc
        raise TypeError(
            "Could not construct ldpc BP+OSD decoder; installed ldpc API did "
            "not match the supported constructor signatures. Last error: "
            f"{last_error}"
        ) from last_error

    def decode(self, syndrome: np.ndarray, max_iter: int | None = None) -> np.ndarray:
        return np.asarray(self.decoder.decode(syndrome.astype(np.uint8)), dtype=np.uint8)


def make_decoder_runner(dem, spec: DecoderSpec, max_iter: int):
    if spec.backend == "ldpc_bp_osd":
        return LdpcBpOsdRunner(dem, max_iter=max_iter, osd_order=spec.osd_order)
    if spec.config is None:
        raise ValueError(f"decoder spec {spec.label!r} has no FlashBP config")
    return flashbp.FlashBP(dem, spec.config)


def logical_prediction(decoder, correction: np.ndarray) -> np.ndarray:
    return (decoder.L @ correction.astype(np.int32)) % 2


def p_token(p: float) -> str:
    return f"{p:.6g}".replace(".", "p").replace("-", "m")


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        return
    if not force:
        reply = input(f"Output dir '{output_dir}' already exists. Overwrite? [y/N]: ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)
    shutil.rmtree(output_dir)


def parse_syndrome_bits(text: str) -> np.ndarray:
    clean = "".join(ch for ch in text if ch in "01")
    if not clean:
        raise ValueError("syndrome bit string must contain at least one 0/1 bit")
    return np.asarray([int(ch) for ch in clean], dtype=np.uint8)


def load_cached_shot(path: str | Path, shot_index: int = 0) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=False)
    syndromes = data["syndromes"]
    if shot_index < 0:
        shot_index += syndromes.shape[0]
    if shot_index < 0 or shot_index >= syndromes.shape[0]:
        raise IndexError(
            f"shot_index {shot_index} outside cache with {syndromes.shape[0]} shots"
        )

    shot = {"syndrome": syndromes[shot_index].astype(np.uint8)}
    for key in (
        "true_obs",
        "true_errors",
        "fail_corrections",
        "success_corrections",
        "fail_pred_obs",
        "success_pred_obs",
        "bp_corrections",
        "ml_corrections",
        "bp_pred_obs",
        "ml_pred_obs",
    ):
        if key in data and len(data[key]) > 0:
            shot[key] = data[key][shot_index].astype(np.uint8)

    if "fail_corrections" in shot and "bp_corrections" not in shot:
        shot["bp_corrections"] = shot["fail_corrections"]
    if "success_corrections" in shot and "ml_corrections" not in shot:
        shot["ml_corrections"] = shot["success_corrections"]
    if "fail_pred_obs" in shot and "bp_pred_obs" not in shot:
        shot["bp_pred_obs"] = shot["fail_pred_obs"]
    if "success_pred_obs" in shot and "ml_pred_obs" not in shot:
        shot["ml_pred_obs"] = shot["success_pred_obs"]

    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"]))
    return shot, metadata


def resolve_code_and_p(args, metadata: dict) -> tuple[str, float]:
    code_name = getattr(args, "code", None) or metadata.get("code") or "steane"
    p = getattr(args, "p", None)
    if p is None:
        p = metadata.get("p", 0.05)
    if code_name not in CODES:
        raise ValueError(
            "could not infer code; pass --code explicitly "
            f"(known: {', '.join(CODES)})"
        )
    return code_name, float(p)


def layout_for_code(code):
    if isinstance(code, BBCode):
        return bb_torus_layout(code.l, code.m)
    if isinstance(code, SurfaceCode):
        return surface_code_layout(code.H_X, code.H_Z, code.d)
    return None


def sample_or_cached_shot(
    dem,
    args,
    cache_path: Path | None,
    shot: dict,
    want_errors: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if getattr(args, "syndrome", None) is not None:
        return parse_syndrome_bits(args.syndrome), None, None
    if "syndrome" in shot:
        return (
            shot["syndrome"].astype(np.uint8),
            shot.get("true_obs"),
            shot.get("true_errors"),
        )

    if getattr(args, "seed", None) is not None:
        np.random.seed(args.seed)
    sampler = dem.compile_sampler()
    if want_errors:
        det, obs, err = sampler.sample(shots=1, return_errors=True)
        return det[0].astype(np.uint8), obs[0].astype(np.uint8), err[0].astype(np.uint8)
    det, obs, _ = sampler.sample(shots=1)
    return det[0].astype(np.uint8), obs[0].astype(np.uint8), None
