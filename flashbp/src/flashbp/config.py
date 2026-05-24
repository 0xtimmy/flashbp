from dataclasses import dataclass


@dataclass
class DecoderConfig:
    decoder:       str        = "simple"   # "simple" | "tensor" | "gbp" | "degree" | "ml" | "surprise_ml"
    degree:        int        = 2          # tensor/degree radius; gbp short_cycles max length
    bond_dim:      int        = 8          # ml: bond dimension chi for TN contraction
    region_policy: str        = "check_neighborhood"  # gbp: "check_neighborhood" | "short_cycles" | active variants
    gbp_backend:   str        = "dense_cpu"  # gbp: "dense_cpu" | "sparse_cpu" | "torch_cuda"
    gbp_max_states:int        = 1 << 22   # gbp: dense states for dense_cpu; valid-state budget for sparse_cpu
    gbp_manual_groups: list | None = None # gbp manual policy: [{"data": [...], "checks": [...], "activation": "always|any|all"}]
    gbp_manual_add_single_checks: bool = True
    gbp_oscillation_boost: float = 1.0    # gbp: multiply region messages after repeated hard-decision states
    gbp_oscillation_boost_cap: float = 64.0
    log:           bool       = False
    log_type:      str        = "simple"   # "simple" | "decode" | "record" | "tensor" | "gbp" | "ml" | "surprise_ml"
    log_level:     int        = 1
    log_console:   bool       = True   # print messages to stdout
    log_file:      str | None = None   # write to file when not None
    log_buffered:  bool       = False  # accumulate until flush() is called
    record_dir:    str | None = None   # directory for animation frames / output
