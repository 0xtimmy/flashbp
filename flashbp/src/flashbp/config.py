from dataclasses import dataclass


@dataclass
class DecoderConfig:
    decoder:      str        = "simple"   # "simple" | "tensor" | "fvs" | "degree" | "ml"
    degree:       int        = 2          # tensor/fvs/degree: neighbourhood radius (>=1)
    bond_dim:     int        = 8          # ml: bond dimension χ for TN contraction
    log:          bool       = False
    log_type:     str        = "simple"   # "simple" | "decode" | "record" | "tensor" | "ml"
    log_level:    int        = 1
    log_console:  bool       = True   # print messages to stdout
    log_file:     str | None = None   # write to file when not None
    log_buffered: bool       = False  # accumulate until flush() is called
    record_dir:   str | None = None   # directory for animation frames / output
