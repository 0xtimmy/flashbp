from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cycles import find_cycles


@dataclass(frozen=True)
class GBPRegionInfo:
    index: int
    policy: str
    center_check: int
    activation: str
    data: tuple[int, ...]
    cycle_checks: tuple[int, ...]
    internal_checks: tuple[int, ...]
    is_fallback: bool = False


class _DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _edges_and_neighbors(H: np.ndarray):
    H = np.asarray(H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    rows, cols = np.nonzero(H)
    edges = [(int(d), int(v)) for d, v in zip(rows, cols)]
    var_checks = [[] for _ in range(num_vars)]
    check_vars = [[] for _ in range(num_checks)]
    for d, v in edges:
        var_checks[v].append(d)
        check_vars[d].append(v)
    return edges, var_checks, check_vars


def _internal_checks(data: set[int], check_vars: list[list[int]]) -> tuple[int, ...]:
    out = []
    for c, vars_for_check in enumerate(check_vars):
        if vars_for_check and all(v in data for v in vars_for_check):
            out.append(c)
    return tuple(out)


def _check_region(
    index: int,
    policy: str,
    check: int,
    check_vars: list[list[int]],
    is_fallback: bool = False,
) -> GBPRegionInfo:
    data = set(check_vars[check])
    return GBPRegionInfo(
        index=index,
        policy=policy,
        center_check=check,
        activation="always",
        data=tuple(sorted(data)),
        cycle_checks=(check,),
        internal_checks=_internal_checks(data, check_vars),
        is_fallback=is_fallback,
    )


def _activation_from_policy(policy: str) -> str:
    if policy.endswith("_any_active") or policy.endswith("_any_on"):
        return "any"
    if policy.endswith("_all_active") or policy.endswith("_all_on"):
        return "all"
    return "always"


def _normalise_policy(policy: str | None) -> tuple[str, bool, str]:
    policy = policy or "check_neighborhood"
    aliases = {
        "check": "check_neighborhood",
        "cycles": "short_cycles",
        "cycles_any_active": "short_cycles_any_active",
        "cycles_any_on": "short_cycles_any_active",
        "short_cycles_any_on": "short_cycles_any_active",
        "cycles_all_active": "short_cycles_all_active",
        "cycles_all_on": "short_cycles_all_active",
        "short_cycles_all_on": "short_cycles_all_active",
        "union_cycles": "short_cycles_union",
        "union_short_cycles": "short_cycles_union",
        "union_cycles_any_active": "short_cycles_union_any_active",
        "union_cycles_any_on": "short_cycles_union_any_active",
        "union_short_cycles_any_active": "short_cycles_union_any_active",
        "short_cycles_union_any_on": "short_cycles_union_any_active",
        "union_cycles_all_active": "short_cycles_union_all_active",
        "union_cycles_all_on": "short_cycles_union_all_active",
        "union_short_cycles_all_active": "short_cycles_union_all_active",
        "short_cycles_union_all_on": "short_cycles_union_all_active",
    }
    policy = aliases.get(policy, policy)
    union = "union" in policy
    if policy == "check_neighborhood":
        family = "check_neighborhood"
    elif "short_cycles" in policy:
        family = "short_cycles"
    else:
        raise ValueError(f"unknown GBP region policy {policy!r}")
    return policy, union, family


def build_gbp_regions(
    H: np.ndarray,
    policy: str = "check_neighborhood",
    degree: int = 2,
    max_axes: int = 32,
) -> list[GBPRegionInfo]:
    """
    Reconstruct the GBP region grouping policy used by the C++ decoder.

    This is intentionally a Python mirror of `gbp_region_policy.cpp`, used for
    visualization and analysis.  It does not participate in decoding.
    """
    H = np.asarray(H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    _, var_checks, check_vars = _edges_and_neighbors(H)
    policy, union_overlaps, family = _normalise_policy(policy)

    if family == "check_neighborhood":
        if degree < 1:
            raise ValueError("check-neighborhood GBP degree must be >= 1")
        regions = []
        for c in range(num_checks):
            data: set[int] = set()
            visited_checks = {c}
            frontier_checks = [c]
            for hop in range(degree):
                new_data = []
                for cc in frontier_checks:
                    for v in check_vars[cc]:
                        if v not in data:
                            data.add(v)
                            new_data.append(v)
                if hop + 1 == degree:
                    break
                new_checks = []
                for v in new_data:
                    for cc in var_checks[v]:
                        if cc not in visited_checks:
                            visited_checks.add(cc)
                            new_checks.append(cc)
                if not new_checks:
                    break
                frontier_checks = new_checks
            if len(data) > max_axes:
                raise RuntimeError(
                    f"check {c} produced {len(data)} axes > max_axes={max_axes}"
                )
            regions.append(
                GBPRegionInfo(
                    index=len(regions),
                    policy=policy,
                    center_check=c,
                    activation="always",
                    data=tuple(sorted(data)),
                    cycle_checks=(c,),
                    internal_checks=_internal_checks(data, check_vars),
                    is_fallback=False,
                )
            )
        return regions

    max_length = degree if degree >= 4 else 8
    activation = _activation_from_policy(policy)
    cycles = find_cycles(H, max_length=max_length)
    seeds: list[tuple[set[int], set[int]]] = []
    check_covered = set()
    for cycle in cycles:
        cycle_checks = {node - num_vars for node in cycle if node >= num_vars}
        if not cycle_checks:
            continue
        data = set()
        for c in cycle_checks:
            if activation == "always":
                check_covered.add(c)
            data.update(check_vars[c])
        seeds.append((cycle_checks, data))

    if union_overlaps and seeds:
        dsu = _DisjointSet(len(seeds))
        owner_check: dict[int, int] = {}
        owner_data: dict[int, int] = {}
        for i, (checks, data) in enumerate(seeds):
            for c in checks:
                if c in owner_check:
                    dsu.unite(i, owner_check[c])
                else:
                    owner_check[c] = i
            for v in data:
                if v in owner_data:
                    dsu.unite(i, owner_data[v])
                else:
                    owner_data[v] = i

        merged: dict[int, tuple[set[int], set[int]]] = {}
        for i, (checks, data) in enumerate(seeds):
            root = dsu.find(i)
            if root not in merged:
                merged[root] = (set(), set())
            merged[root][0].update(checks)
            merged[root][1].update(data)
        seeds = list(merged.values())

    regions: list[GBPRegionInfo] = []
    for checks, data in seeds:
        if len(data) > max_axes:
            raise RuntimeError(
                f"cycle region produced {len(data)} axes > max_axes={max_axes}"
            )
        regions.append(
            GBPRegionInfo(
                index=len(regions),
                policy=policy,
                center_check=min(checks),
                activation=activation,
                data=tuple(sorted(data)),
                cycle_checks=tuple(sorted(checks)),
                internal_checks=_internal_checks(data, check_vars),
                is_fallback=False,
            )
        )

    add_all_single_check_regions = activation != "always"
    for c in range(num_checks):
        if not add_all_single_check_regions and c in check_covered:
            continue
        region = _check_region(len(regions), policy, c, check_vars, is_fallback=True)
        if len(region.data) > max_axes:
            raise RuntimeError(
                f"fallback check region {c} produced {len(region.data)} axes "
                f"> max_axes={max_axes}"
            )
        regions.append(region)

    return regions


def region_is_active(region: GBPRegionInfo, syndrome: np.ndarray) -> bool:
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    if region.activation == "always":
        return True
    if region.activation == "any":
        return any(bool(syndrome[c]) for c in region.cycle_checks)
    if region.activation == "all":
        return bool(region.cycle_checks) and all(
            bool(syndrome[c]) for c in region.cycle_checks
        )
    raise ValueError(f"unknown activation mode {region.activation!r}")


def active_gbp_regions(
    regions: list[GBPRegionInfo],
    syndrome: np.ndarray,
) -> list[GBPRegionInfo]:
    return [region for region in regions if region_is_active(region, syndrome)]
