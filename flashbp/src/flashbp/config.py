from dataclasses import dataclass


@dataclass
class DecoderConfig:
    decoder:       str        = "simple"   # "simple" | "tensor" | "gbp" | "degree" | "ml" | "surprise_ml"
    degree:        int        = 2          # tensor/degree radius; gbp short_cycles max length
    bond_dim:      int        = 8          # ml: bond dimension chi for TN contraction
    region_policy: str        = "check_neighborhood"  # gbp: "check_neighborhood" | "short_cycles" | active variants
    log:           bool       = False
    log_type:      str        = "simple"   # "simple" | "decode" | "record" | "tensor" | "ml" | "surprise_ml"
    log_level:     int        = 1
    log_console:   bool       = True   # print messages to stdout
    log_file:      str | None = None   # write to file when not None
    log_buffered:  bool       = False  # accumulate until flush() is called
    record_dir:    str | None = None   # directory for animation frames / output
