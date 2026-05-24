python evaluations/threshold_sweep.py \
    --code hbb \
    --decoders simple,bposd,gbp:4:short_cycles_all_active,gbp:4:short_cycles_union_all_active:sparse,gbp:4:short_cycles_any_active,gbp:4:short_cycles_union_any_active:sparse,gbp:2:short_cycles_all_active,gbp:2:short_cycles_union_all_active:sparse,gbp:2:short_cycles_any_active,gbp:2:short_cycles_union_any_active:sparse \
    --shots 100000 --p-max 0.01 --p-min 0.0005 \
    --num-p 4