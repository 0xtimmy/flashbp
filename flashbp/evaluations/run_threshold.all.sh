python evaluations/threshold_sweep.py \
    --code steane \
    --decoders simple,bposd,ml,gbp:2:short_cycles_any_active,gbp:2:short_cycles_all_active,gbp:2:short_cycles_union_any_active,gbp:2:short_cycles_union_all_active \
    --shots 1000 --p-max 0.1 --p-min 0.01