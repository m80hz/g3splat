#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ScanNet depth evaluation indices with 3, 8, 12, and 24 "
            "context views from a 2-context input index."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input evaluation index JSON (e.g., evaluation_index_scannetv1_depth.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write output JSON files. "
            "Defaults to the input file's directory."
        ),
    )
    parser.add_argument(
        "--max-dist",
        type=int,
        default=2,
        help=(
            "Maximum allowed index distance between context views and target "
            "views. Entries violating this are skipped. Default: 2."
        ),
    )
    parser.add_argument(
        "--num-context",
        type=int,
        nargs="*",
        default=[3, 8, 12, 24],
        help=(
            "List of context-view counts to generate variants for. "
            "Default: 3 8 12 24."
        ),
    )
    return parser.parse_args()


def is_valid_entry(context: List[int], target: List[int], max_dist: int) -> bool:
    """Check that every target is within max_dist of at least one context index.

    This now measures distance w.r.t. the *new* context set only and is used
    as a soft filter after we resample contexts around the original ones.
    """

    if not context:
        return False

    for t in target:
        if all(abs(t - c) > max_dist for c in context):
            return False
    return True


def sample_uniform_inclusive(start: int, end: int, k: int) -> List[int]:
    """Sample k integers uniformly (deterministic grid) from [start, end].

    This returns a monotonically increasing list including both end-points when
    possible and without duplicates. If k == 1, returns the midpoint.
    """

    if k <= 0:
        return []

    if start == end:
        return [start] * k

    length = end - start
    if k == 1:
        return [start + length // 2]

    if k >= length + 1:
        # Saturate to full range
        return list(range(start, end + 1))

    step = length / (k - 1)
    sampled = []
    for i in range(k):
        v = int(round(start + i * step))
        v = max(start, min(end, v))
        if not sampled or v != sampled[-1]:
            sampled.append(v)

    # If we ended up with fewer due to rounding collisions, pad greedily
    cur = start
    while len(sampled) < k and cur <= end:
        if cur not in sampled:
            sampled.append(cur)
        cur += 1

    sampled.sort()
    return sampled


def build_context_indices(
    context: List[int], target: List[int], desired_k: int, max_dist: int
) -> List[int]:
    """Build a new set of context indices.

    Rules (as interpreted from the user request):
    - Original input has 2 context views: [c0, c1] with c0 < c1.
    - We never sample context indices greater than c1.
    - We only sample indices *before* the first context view.
    - We keep the number of target views unchanged (handled outside).
    - We enforce that all targets stay within max_dist of at least one
      context view; otherwise, the scene is skipped externally.
    """

    if len(context) < 1:
        return []

    c_min = min(context)
    c_max = max(context)

    # Expand a band around [c_min, c_max] using max_dist as "radius".
    band_start = max(0, c_min - max_dist)
    band_end = c_max + max_dist

    if band_start > band_end or desired_k <= 0:
        return []

    # We ignore the exact target positions here and simply sample desired_k
    # indices uniformly from the band around the original contexts.
    max_possible = band_end - band_start + 1
    k = min(desired_k, max_possible)

    # Initial uniform sampling in the band
    ctx = sample_uniform_inclusive(band_start, band_end, k)

    # Enforce that context indices are disjoint from target indices.
    target_set = set(target)
    ctx = [c for c in ctx if c not in target_set]

    # If we removed too many due to collisions, try to refill from nearby
    # indices within the band that are not targets.
    if len(ctx) < k:
        used = set(ctx)
        for c in range(band_start, band_end + 1):
            if c in target_set or c in used:
                continue
            ctx.append(c)
            used.add(c)
            if len(ctx) >= k:
                break

    # If we still have no contexts after avoiding targets, give up for this
    # entry and let the caller skip it.
    if not ctx:
        return []

    return sorted(ctx)


def generate_variants(
    entries: List[Dict[str, Any]], num_context_list: List[int], max_dist: int
) -> Dict[int, List[Dict[str, Any]]]:
    """Generate per-K context variants from base entries."""

    variants = {k: [] for k in num_context_list}

    for entry in entries:
        scene = entry["scene"]
        base_context = entry["context"]
        target = entry["target"]

        for k in num_context_list:
            new_context = build_context_indices(base_context, target, k, max_dist)
            if not new_context:
                # Skip this scene for this K.
                continue

            # Sanity: all targets within max_dist of at least one new context.
            if not is_valid_entry(new_context, target, max_dist):
                continue

            variants[k].append(
                {
                    "scene": scene,
                    "context": new_context,
                    "target": target,
                }
            )

    return variants


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir or input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r") as f:
        entries = json.load(f)

    num_context_list = sorted(set(args.num_context))

    variants = generate_variants(entries, num_context_list, args.max_dist)

    stem = input_path.stem
    wrote_any = False
    for k in num_context_list:
        out_entries = variants.get(k, [])
        if not out_entries:
            print(f"No valid entries for K = {k}")
            continue
        out_path = output_dir / f"{stem}_ctx{k}.json"
        with out_path.open("w") as f:
            json.dump(out_entries, f, indent=4)
        print(f"Wrote {len(out_entries)} entries to {out_path}")
        wrote_any = True

    if not wrote_any:
        print(
            "No output files were generated. "
            "All entries may have been filtered by the constraints."
        )


if __name__ == "__main__":
    main()
